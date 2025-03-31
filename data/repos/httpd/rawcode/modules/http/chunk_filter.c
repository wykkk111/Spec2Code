/* Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * chunk_filter.c --- HTTP/1.1 chunked transfer encoding filter.
 */

#include "apr_strings.h"
#include "apr_lib.h"
#include "apr_thread_proc.h"    /* for RLIMIT stuff */

#define APR_WANT_STRFUNC
#include "apr_want.h"

#include "httpd.h"
#include "http_config.h"
#include "http_connection.h"
#include "http_core.h"
#include "http_log.h"
#include "http_protocol.h"  /* For index_of_response().  Grump. */
#include "http_request.h"

#include "util_filter.h"
#include "util_ebcdic.h"
#include "ap_mpm.h"
#include "scoreboard.h"

#include "mod_core.h"


APLOG_USE_MODULE(http);


typedef struct chunk_out_ctx {
    int bad_gateway_seen;
    apr_table_t *trailers;
} chunk_out_ctx;


apr_status_t ap_http_chunk_filter(ap_filter_t *f, apr_bucket_brigade *b)
{
    conn_rec *c = f->r->connection;
    chunk_out_ctx *ctx = f->ctx;
    apr_bucket_brigade *more, *tmp;
    apr_bucket *e;
    apr_status_t rv;

    if (!ctx) {
        ctx = f->ctx = apr_pcalloc(f->r->pool, sizeof(*ctx));
    }

    for (more = tmp = NULL; b; b = more, more = NULL) {
        apr_off_t bytes = 0;
        apr_bucket *eos = NULL; /* EOS bucket, or FLUSH preceding EOS */
        apr_bucket *flush = NULL;
        /* XXX: chunk_hdr must remain at this scope since it is used in a
         *      transient bucket.
         */
        char chunk_hdr[20]; /* enough space for the snprintf below */


        for (e = APR_BRIGADE_FIRST(b);
             e != APR_BRIGADE_SENTINEL(b);
             e = APR_BUCKET_NEXT(e))
        {
            if (APR_BUCKET_IS_METADATA(e)) {
                if (APR_BUCKET_IS_EOS(e)) {
                    /* there shouldn't be anything after the eos */
                    ap_log_rerror(APLOG_MARK, APLOG_TRACE2, 0, f->r,
                                  "ap_http_chunk_filter eos seen, removing filter");
                    ap_remove_output_filter(f);
                    eos = e;
                    break;
                }
                if (AP_BUCKET_IS_ERROR(e) &&
                    (((ap_bucket_error *)(e->data))->status == HTTP_BAD_GATEWAY ||
                     ((ap_bucket_error *)(e->data))->status == HTTP_GATEWAY_TIME_OUT)) {
                    /*
                     * We had a broken backend. Memorize this in the filter
                     * context.
                     */
                    ap_log_rerror(APLOG_MARK, APLOG_TRACE2, 0, f->r,
                                  "ap_http_chunk_filter bad gateway error, suppressing end chunk");
                    ctx->bad_gateway_seen = 1;
                    continue;
                }
                if (APR_BUCKET_IS_FLUSH(e)) {
                    flush = e;

                    /* Special case to catch common brigade ending of
                     * [FLUSH] [EOS] - insert the last_chunk before
                     * the FLUSH rather than between the FLUSH and the
                     * EOS. */
                    if (e != APR_BRIGADE_LAST(b)) {
                        if (APR_BUCKET_IS_EOS(APR_BUCKET_NEXT(e))) {
                            eos = e;
                            /* anything after EOS is dropped, no need
                             * to split. */
                        }
                        else {
                            more = apr_brigade_split_ex(b, APR_BUCKET_NEXT(e), tmp);
                        }
                    }
                    break;
                }
                if (AP_BUCKET_IS_HEADERS(e)) {
                    ap_bucket_headers *hdrs = e->data;
                    if (!apr_is_empty_table(hdrs->headers)) {
                        if (!ctx->trailers) {
                            ctx->trailers = apr_table_make(f->r->pool, 5);
                        }
                        apr_table_overlap(ctx->trailers, hdrs->headers, APR_OVERLAP_TABLES_MERGE);
                    }
                }
            }
            else if (e->length == (apr_size_t)-1) {
                /* unknown amount of data (e.g. a pipe) */
                const char *data;
                apr_size_t len;

                rv = apr_bucket_read(e, &data, &len, APR_BLOCK_READ);
                if (rv != APR_SUCCESS) {
                    return rv;
                }
                if (len > 0) {
                    /*
                     * There may be a new next bucket representing the
                     * rest of the data stream on which a read() may
                     * block so we pass down what we have so far.
                     */
                    bytes += len;
                    more = apr_brigade_split_ex(b, APR_BUCKET_NEXT(e), tmp);
                    break;
                }
                else {
                    /* If there was nothing in this bucket then we can
                     * safely move on to the next one without pausing
                     * to pass down what we have counted up so far.
                     */
                    continue;
                }
            }
            else {
                bytes += e->length;
            }
        }

        /*
         * XXX: if there aren't very many bytes at this point it may
         * be a good idea to set them aside and return for more,
         * unless we haven't finished counting this brigade yet.
         */
        /* if there are content bytes, then wrap them in a chunk */
        if (bytes > 0) {
            apr_size_t hdr_len;
            /*
             * Insert the chunk header, specifying the number of bytes in
             * the chunk.
             */
            ap_log_rerror(APLOG_MARK, APLOG_TRACE2, 0, f->r,
                          "ap_http_chunk_filter sending chunk of %"
                          APR_UINT64_T_HEX_FMT " bytes", (apr_uint64_t)bytes);
            hdr_len = apr_snprintf(chunk_hdr, sizeof(chunk_hdr),
                                   "%" APR_UINT64_T_HEX_FMT CRLF, (apr_uint64_t)bytes);
            ap_xlate_proto_to_ascii(chunk_hdr, hdr_len);
            e = apr_bucket_transient_create(chunk_hdr, hdr_len,
                                            c->bucket_alloc);
            APR_BRIGADE_INSERT_HEAD(b, e);

            /*
             * Insert the end-of-chunk CRLF before an EOS or
             * FLUSH bucket, or appended to the brigade
             */
            e = apr_bucket_immortal_create(CRLF_ASCII, 2, c->bucket_alloc);
            if (flush != NULL) {
                APR_BUCKET_INSERT_BEFORE(flush, e);
            }
            else if (eos != NULL) {
                APR_BUCKET_INSERT_BEFORE(eos, e);
            }
            else {
                APR_BRIGADE_INSERT_TAIL(b, e);
            }
        }

        /* RFC 2616, Section 3.6.1
         *
         * If there is an EOS bucket, then prefix it with:
         *   1) the last-chunk marker ("0" CRLF)
         *   2) the trailer
         *   3) the end-of-chunked body CRLF
         *
         * We only do this if we have not seen an error bucket with
         * status HTTP_BAD_GATEWAY. We have memorized an
         * error bucket that we had seen in the filter context.
         * The error bucket with status HTTP_BAD_GATEWAY indicates that the
         * connection to the backend (mod_proxy) broke in the middle of the
         * response. In order to signal the client that something went wrong
         * we do not create the last-chunk marker and set c->keepalive to
         * AP_CONN_CLOSE in the core output filter.
         *
         * XXX: it would be nice to combine this with the end-of-chunk
         * marker above, but this is a bit more straight-forward for
         * now.
         */
        if (eos && !ctx->bad_gateway_seen) {
            ap_h1_add_end_chunk(b, eos, f->r, ctx->trailers);
        }

        /* pass the brigade to the next filter. */
        rv = ap_pass_brigade(f->next, b);
        apr_brigade_cleanup(b);
        if (rv != APR_SUCCESS || eos != NULL) {
            return rv;
        }
        tmp = b;
    }
    return APR_SUCCESS;
}
