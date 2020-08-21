/* 
 * "Restless reachability in temporal graphs: algebraic methods and applications"
 * 
 * This experimental source code is supplied to accompany the 
 * aforementioned paper. 
 * 
 * The source code is configured for a gcc build to a native 
 * microarchitecture that must support the AVX2 and PCLMULQDQ 
 * instruction set extensions. Other builds are possible but 
 * require manual configuration of 'Makefile' and 'builds.h'.
 * 
 * The source code is subject to the following license.
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2020 Anonymous authors.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include<sys/utsname.h>
#include<string.h>
#include<stdarg.h>
#include<assert.h>
#include<math.h>

#include<omp.h>

typedef long int index_t;

#include"ffprng.h"

#define UNDEFINED -1
#define BUILD_PARALLEL

/******************************************************* Common subroutines. */

#define ERROR(...) error(__FILE__,__LINE__,__func__,__VA_ARGS__);

static void error(const char *fn, int line, const char *func, 
                  const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, 
            "ERROR [file = %s, line = %d]\n"
            "%s: ",
            fn,
            line,
            func);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();    
}

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}

/******************************************************** Memory allocation. */

#define MALLOC(id, x) malloc_wrapper(id, x)
#define FREE(id, x) free_wrapper(id, x)

index_t malloc_balance = 0;

void *malloc_wrapper(const char *id, size_t size)
{
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;
#ifdef MEM_INVENTORY
    fprintf(stdout, "alloc: %10s %7.3lf GiB\n", id, inGiB(size));
    fflush(stdout);
#endif
    return p;
}

void free_wrapper(const char *id, void *p)
{
    free(p);
    malloc_balance--;
#ifdef MEM_INVENTORY
    fprintf(stdout, "free: %10s\n", id);
    fflush(stdout);
#endif
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC("t", sizeof(index_t)*n);
    return t;
}

/****************************************************************** sorting. */

void shellsort(index_t n, index_t *a)
{
    index_t h = 1;
    index_t i;
    for(i = n/3; h < i; h = 3*h+1)
        ;
    do {
        for(i = h; i < n; i++) {
            index_t v = a[i];
            index_t j = i;
            do {
                index_t t = a[j-h];
                if(t <= v)
                    break;
                a[j] = t;
                j -= h;
            } while(j >= h);
            a[j] = v;
        }
        h /= 3;
    } while(h > 0);
}

#ifdef DEBUG
void print_array(const char *name, index_t n, index_t *a, index_t offset)
{
    fprintf(stdout, "%s (%ld):", name, n);
    for(index_t i = 0; i < n; i++) {
        fprintf(stdout, " %ld", a[i] == -1 ? -1 : a[i]+offset);
    }
    fprintf(stdout, "\n");
}
#endif


/************************************************* Random number generators. */

index_t irand(void)
{
    return (((index_t) rand())<<31)^((index_t) rand());
}

void rand_nums(index_t n, index_t *p, index_t seed)
{
    srand(seed);
    for(index_t i = 0; i < n; i++)
        p[i] = irand();
}

void randseq_range(index_t n, index_t range, index_t *p, index_t seed)
{
#ifdef BUILD_PARALLEL
    index_t nt = 64;
#else
    index_t nt = 1;
#endif
    ffprng_t base;
    FFPRNG_INIT(base, seed);
    index_t block_size = n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t rs = (index_t) (rnd&0X7FFFFFFFFFFFFFFF);
            p[i] = rs%range;
        }
    }
}

void randshuffle_seq(index_t n, index_t *p, ffprng_t gen)
{
    for(index_t i = 0; i < n-1; i++) {
        ffprng_scalar_t rnd;
        FFPRNG_RAND(rnd, gen);     
        index_t x = i+(rnd%(n-i));
        index_t t = p[x];
        p[x] = p[i];
        p[i] = t;
    }
}

void randperm(index_t n, index_t *p, index_t seed)
{
#ifdef BUILD_PARALLEL
    index_t nt = 64;
#else
    index_t nt = 1;
#endif
    index_t block_size = n/nt;
    index_t f[128][128];
    assert(nt < 128);

    ffprng_t base;
    FFPRNG_INIT(base, seed);    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        for(index_t j = 0; j < nt; j++)
            f[t][j] = 0;        
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        ffprng_t gen;
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            f[t][bin]++;
        }
    }    

    for(index_t bin = 0; bin < nt; bin++) {
        for(index_t t = 1; t < nt; t++) {
            f[0][bin] += f[t][bin];
        }
    }
    index_t run = 0;
    for(index_t j = 1; j <= nt; j++) {
        index_t fp = f[0][j-1];
        f[0][j-1] = run;
        run += fp;
    }
    f[0][nt] = run;

    FFPRNG_INIT(base, seed);    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = 0;
        index_t stop = n-1;
        index_t pos = f[0][t];
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            index_t bin = (index_t) ((unsigned long) rnd)%((unsigned long)nt);
            if(bin == t)
                p[pos++] = i;
        }
        assert(pos == f[0][t+1]);
    }

    FFPRNG_INIT(base, (seed^0x9078563412EFDCABL));    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t fwd, gen;
        index_t start = f[0][t];
        index_t stop = f[0][t+1]-1;
        index_t u;
        FFPRNG_FWD(fwd, (1234567890123456L*t), base);
        FFPRNG_RAND(u, fwd);
        FFPRNG_INIT(gen, u);
        randshuffle_seq(stop-start+1, p + start, gen);
    }
}

index_t *alloc_randperm(index_t n, index_t seed)
{
    index_t *p = alloc_idxtab(n);
    randperm(n, p, seed);
    return p;
}

/************************************************ Rudimentary graph builder. */
enum graph_type{SMALL, REGULAR, NOMOTIF, POWLAW, CLIQUE, DENDOGRAM};

typedef struct 
{
    enum graph_type gtype;
    index_t is_directed;
    index_t num_vertices;
    index_t num_edges;
    index_t time_limit;
    index_t edge_capacity;
    index_t *edges;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC("a", sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE("was", was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);
    graph_t *g       = (graph_t *) MALLOC("g", sizeof(graph_t));
    g->is_directed   = 0; // default: undirected graph
    g->num_vertices  = n;
    g->num_edges     = 0;
    g->time_limit    = 0;
    g->edge_capacity = 1000;
    g->edges         = enlarge(3*g->edge_capacity, 0, (void *) 0);

    return g;
}

void graph_set_is_directed(graph_t *g, index_t is_dir)
{
    assert(is_dir == 0 || is_dir == 1);
    g->is_directed = is_dir;
}

void graph_free(graph_t *g)
{
    FREE("g->edges", g->edges);
    FREE("g", g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v, index_t t)
{
    assert(u >= 0 && 
           v >= 0 && 
           u < g->num_vertices &&
           v < g->num_vertices);

    assert(t >= 0 && t < g->time_limit);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(6*g->edge_capacity, 6*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 3*g->num_edges;
    g->num_edges++;
    e[0] = u;
    e[1] = v;
    e[2] = t;
    //shellsort(2, e);
}

index_t *graph_edgebuf(graph_t *g, index_t cap)
{
    //assert(g->has_target == 0);
    g->edges = enlarge(3*g->edge_capacity+3*cap, 3*g->edge_capacity, g->edges);
    index_t *e = g->edges + 3*g->num_edges;
    g->edge_capacity += cap;
    g->num_edges += cap;
    return e;
}

/************************************************************ Output graph. */

void graph_out_dimacs(FILE *out, graph_t *g, index_t max_rt)
{
    index_t n       = g->num_vertices;
    index_t m       = g->num_edges;
    //index_t k       = g->motif_size;
    index_t max_t   = g->time_limit;
    //index_t max_rt  = g->rest_limit;
    index_t *e      = g->edges;
    fprintf(out, "p motif %ld %ld %ld %ld %ld\n", 
                 (long) n, (long) m, (long) max_t, (long) max_rt, 
                 (long) g->is_directed);

    // output edge list
    for(index_t i = 0; i < m; i++) {
        index_t *e_i = e + 3*i;
        index_t u = e_i[0]+1;
        index_t v = e_i[1]+1;
        index_t t = e_i[2]+1;
        fprintf(out, "e %ld %ld %ld\n", u, v, t);
    }
}

void graph_out_rest_time(FILE *out, index_t n, index_t *rest_time)
{
    for(index_t u = 0; u < n; u++)
        fprintf(stdout, "r %ld %ld\n", u+1, rest_time[u]+1);
}

void graph_out_sources(FILE *out, index_t num_sources, index_t *sources)
{
    if(num_sources == 0) return;

    fprintf(out, "s %ld", num_sources);
    for(index_t i = 0; i < num_sources; i++)
        fprintf(out, " %ld", sources[i]==UNDEFINED ? sources[i] : sources[i]+1);
    fprintf(out, "\n");
}

void graph_out_separators(FILE *out, index_t num_separators, index_t *separators)
{
    if(num_separators == 0) return;

    fprintf(out, "t %ld", num_separators);
    for(index_t i = 0; i < num_separators; i++)
        fprintf(out, " %ld", separators[i]==UNDEFINED ? separators[i] : separators[i]+1);
    fprintf(out, "\n");
}



index_t *graph_degree_dist(graph_t *g)
{
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t *deg = alloc_idxtab(n);
    for(index_t i = 0; i < n; i++)
        deg[i] = 0;
    for(index_t j = 0; j < m; j++) {
        deg[g->edges[2*j]]++;
        deg[g->edges[2*j+1]]++;
    }
    return deg;
}

void print_stat(FILE *out, graph_t *g) 
{
    index_t n = g->num_vertices;
    index_t *deg = graph_degree_dist(g);
    for(index_t i = 0; i < n; i++) {
        fprintf(out, "deg[%ld] = %ld\n", i, deg[i]);
    }
    FREE("deg", deg);
}

/***************** A quick-and-dirty generator for a power law distribution. */

double beta(index_t n, index_t k)
{
    double min, mid, max;
    index_t niter = 200;
    min = 1.0;
    max = exp((log(n)/(double) (k-1)));
    for(index_t i = 0; i < niter; i++) {
        mid = (min+max)/2.0;
        double nn = (1-pow(mid,k))/(1-mid);
        if(nn < n)
            min = mid;
        if(nn > n)
            max = mid;
    }
    return mid;
}

double degnormalizer(index_t n, index_t d, index_t *freq, index_t k, double alpha)
{
    double fpowasum = 0.0;
    for(index_t i = 0; i < k; i++)
        fpowasum += pow(freq[i],alpha+1);
    return log(d*n)-log(fpowasum);
}

void mkpowlawdist(index_t *deg, index_t *freq, 
                  index_t n, index_t d, double alpha, index_t k)
{

    double b = beta(n, k);
    index_t fsum = 0;
    for(index_t j = 0; j < k; j++) {
        freq[j] = round((1-pow(b,j+1))/(1-b)-fsum);
        fsum += freq[j];
    }
    double dn = degnormalizer(n, d, freq, k, alpha);

    double t = 0.0;
    index_t dfsum = 0;
    for(index_t j = 0; j < k; j++) {
        t += exp(dn)*pow(freq[j],alpha+1);
        double tt = t-dfsum;
        deg[j] = round(tt/freq[j]);
        dfsum += deg[j]*freq[j];
    }


    if(dfsum % 2 == 1) {
        index_t i = k-1;
        for(; i >= 0; i--) {
            if(deg[i] % 2 == 1 &&
               freq[i] % 2 == 1) {
                freq[i]++;
                dfsum += deg[i];
                break;
            }
        }
        assert(i >= 0);
    }

    fprintf(stderr, 
            "powlaw: n = %ld, d = %ld, alpha = %lf, w = %ld, beta = %lf, norm = %lf\n", 
            n, d, alpha, k, b, dn);
}


/************************************************** Test graph generator(s). */

/* Generators for instances with a unique match. */

/* Use the bits of idx to plant a path on the corresponding
 * vertices. Uses the n least significant bits of idx. */

/*
void graph_set_rand_target(graph_t *g, index_t k, index_t max_color, 
                           index_t num_targets, index_t seed) 
{
    g->has_target  = 1;
    g->motif_size  = k;
    g->num_targets = num_targets;
    g->target     = (index_t *) MALLOC("g->target", sizeof(index_t)*k);
    randseq_range(k, max_color, g->target, seed);

    g->num_colors = max_color;
    randseq_range(g->num_vertices, max_color, g->colors, seed);
    index_t *vertex_shuffle = alloc_randperm(g->num_vertices, seed);
    g->motif_counts = (index_t *) MALLOC("g->motif_counts", sizeof(index_t)*num_targets*k);


#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < num_targets; i++) {
        for(index_t j = 0; j < k; j++) {
            index_t u = vertex_shuffle[i*k+j];
            g->motif_counts[i*k+j] = u;
            g->colors[u] = g->target[j];
        }
    }
    FREE("vertex_shuffle", vertex_shuffle);
}
*/

void get_rest_time_const(index_t n, index_t max_rt, index_t **rest_time_out)
{
    index_t *rest_time = (index_t *) MALLOC("rest time", n*sizeof(index_t));
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        rest_time[u] = max_rt-1;

    *rest_time_out = rest_time;
}

void get_rest_time_random(index_t n, index_t max_rt, index_t seed, index_t **rest_time_out)
{
    index_t *rest_time = (index_t *) MALLOC("rest time", n*sizeof(index_t));
    randseq_range(n, max_rt, rest_time, seed);
    *rest_time_out = rest_time;
}

void get_random_vertices(index_t k, index_t n, index_t seed, index_t **vertices_out)
{
    //index_t *vertex_shuffle = (index_t *) MALLOC("vertex shuffle", n*sizeof(index_t));
    index_t *vertex_shuffle = alloc_randperm(n, seed);
    index_t *vertices = (index_t *) MALLOC("vertices", k*sizeof(index_t));

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < k; i++) {
        vertices[i] = vertex_shuffle[i];
    }
    FREE("vertex_shuffle", vertex_shuffle);

    *vertices_out = vertices;
}


/* The configuration model. 
 * Warning: no check for repeated edges and loops. */

graph_t *graph_config_rand(index_t n, index_t ndd, 
                           index_t max_time, index_t max_resttime,
                           index_t *degree, index_t *freq, index_t seed) 
{
    index_t num_incidences = 0;
    for(index_t j = 0; j < ndd; j++)
        num_incidences += degree[j]*freq[j];
    assert(num_incidences % 2 == 0);

    index_t *vertex_id = alloc_idxtab(num_incidences);
    index_t pos = 0;
    index_t vno = 0;
    for(index_t j = 0; j < ndd; j++) {
        for(index_t k = 0; k < freq[j]; k++) {
            for(index_t l = 0; l < degree[j]; l++)
                vertex_id[pos++] = vno;
            vno++;
        }
    }
    index_t *vertex_shuffle = alloc_randperm(n, seed);
    index_t *incidence_shuffle = alloc_randperm(num_incidences, seed);
    index_t *time_shuffle = (index_t *) MALLOC("time_shuffle", sizeof(index_t)*num_incidences/2);
    randseq_range(num_incidences/2, max_time, time_shuffle, seed);

    graph_t *g = graph_alloc(n);
    index_t *e = graph_edgebuf(g, num_incidences/2);
    g->time_limit = max_time;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < num_incidences/2; i++) {
        index_t u = vertex_shuffle[vertex_id[incidence_shuffle[2*i]]];
        index_t v = vertex_shuffle[vertex_id[incidence_shuffle[2*i+1]]];
        index_t t = time_shuffle[i];
        //if(u > v) {
        //    index_t temp = u;
        //    u = v;
        //    v = temp;
        //}
        e[3*i]   = u;
        e[3*i+1] = v;
        e[3*i+2] = t;
    }

    FREE("vertex_id", vertex_id);
    FREE("vertex_shuffle", vertex_shuffle);
    FREE("incidence_shuffle", incidence_shuffle);
    FREE("time_shuffle", time_shuffle);
    return g;
}

/*
graph_t *graph_config_rand_test(index_t n, index_t ndd, index_t max_time, 
                                index_t max_resttime, index_t num_targets,
                                index_t *degree, index_t *freq, index_t seed) 
{
    index_t num_incidences = 0;
    for(index_t j = 0; j < ndd; j++)
        num_incidences += degree[j]*freq[j];
    assert(num_incidences % 2 == 0);

    index_t *vertex_id = alloc_idxtab(num_incidences);
    index_t pos = 0;
    index_t vno = 0;
    for(index_t j = 0; j < ndd; j++) {
        for(index_t k = 0; k < freq[j]; k++) {
            for(index_t l = 0; l < degree[j]; l++)
                vertex_id[pos++] = vno;
            vno++;
        }
    }
    index_t *vertex_shuffle = alloc_randperm(n, seed);
    index_t *incidence_shuffle = alloc_randperm(num_incidences, seed);
    index_t *time_shuffle = (index_t *) MALLOC("time_shuffle", sizeof(index_t)*num_incidences/2);
    randseq_range(num_incidences/2, max_time, time_shuffle, seed);

    graph_t *g = graph_alloc(n + num_targets);
    index_t *e = graph_edgebuf(g, num_incidences/2);
    g->time_limit = max_time;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < num_incidences/2; i++) {
        index_t u = vertex_shuffle[vertex_id[incidence_shuffle[2*i]]];
        index_t v = vertex_shuffle[vertex_id[incidence_shuffle[2*i+1]]];
        index_t t = time_shuffle[i];
        if(u > v) {
            index_t temp = u;
            u = v;
            v = temp;
        }
        e[3*i]   = u;
        e[3*i+1] = v;
        e[3*i+2] = t;
    }

    // initialise resting times at random
    //g->rest_limit = max_resttime;
    randseq_range(g->num_vertices, max_resttime, g->rest_time, seed);


    FREE("vertex_id", vertex_id);
    FREE("vertex_shuffle", vertex_shuffle);
    FREE("incidence_shuffle", incidence_shuffle);
    FREE("time_shuffle", time_shuffle);
    return g;
}
*/

graph_t *graph_reg_rand(index_t n, index_t d, index_t max_t, index_t max_rt, index_t seed)
{
    index_t deg[128];
    index_t freq[128];
    deg[0] = d;
    freq[0] = n;
    return graph_config_rand(n, 1, max_t, max_rt, deg, freq, seed);
}


graph_t *graph_powlaw_rand(index_t n, index_t d, index_t max_t, index_t max_rt,
                           double alpha, index_t w, index_t seed)
{
    index_t *deg = (index_t *) MALLOC("deg", w*sizeof(index_t));
    index_t *freq = (index_t *) MALLOC("freq", w*sizeof(index_t));
    mkpowlawdist(deg, freq, n, d, alpha, w);
    graph_t *g = graph_config_rand(n, w, max_t, max_rt, deg, freq, seed);
    FREE("deg", deg);
    FREE("freq", freq);
    return g;
}


/*
graph_t *graph_reg_rand_test(index_t n, index_t d, index_t max_t, 
                             index_t max_rt, index_t num_targets, index_t seed)
{
    index_t deg[128];
    index_t freq[128];
    deg[0] = d;
    freq[0] = n;
    return graph_config_rand_test(n, 1, max_t, max_rt, num_targets, deg, freq, seed);
}
*/

/****************************************************** Program entry point. */

int main(int argc, char **argv)
{
    if(argc < 2 || !strcmp(argv[argc-1], "-h")) {
        fprintf(stdout, 
            "usage: %s <type> <arguments>\n"
            "available types (all parameters positive integers unless indicated otherwise):\n"
            "\n"
            "  regular         <n> <d> <t> <rt> <ns> <nt> <seed>               (with 1 <= k <= n and n*d even)\n"
            "  regular-const   <n> <d> <t> <rt> <ns> <nt> <seed>               (with 1 <= k <= n and n*d even)\n"
            "  powlaw          <n> <d> <al> <w> <t> <rt> <ns> <nt> <seed>      (with al < 0.0, 2 <= w <= n, 1 <= k <= n)\n"
            "  powlaw-const    <n> <d> <al> <w> <t> <rt> <ns> <nt> <seed>      (with al < 0.0, 2 <= w <= n, 1 <= k <= n)\n"
            "\n"
            "-----------------------------------------------------------------------------\n"
            "\tArguments\n"
            "\t<n>    : number of vertices, integer value 1 <= n <= 2^63\n"
            "\t<d>    : degree, n*d even\n"
            "\t<t>    : maximum timestamp, integer value 1 <= n <= 2^63\n"
            "\t<rt>   : maximum resting time, integer value 1 <= rt <= t\n"
            "\t<ns>   : number of sources, integer value in range 1 to <n>\n"
            "\t<nt>   : number of separators, integer value in range 1 to <n>\n"
            "\t<al>   : alpha, float value < 0.0\n"
            "\t<w>    : weight, integer value 2 <= w <= n\n"
            "\t<seed> : seed for random number generator, integer value 1 <= seed <= 2^63\n"
            ,
            argv[0]);
        return 0;
    }

    fprintf(stderr, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stderr, " %s", argv[f]);
    fprintf(stderr, "\n");    

    graph_t *g = (graph_t *) 0;

    char *type = argv[1];

    index_t n               = 0;
    index_t d               = 0;
    index_t max_t           = 0;
    index_t max_rt          = 0;
    index_t num_sources     = 0;
    index_t num_separators  = 0;
    index_t seed            = 0;
    double al               = 0.0;
    index_t w               = 0;

    // d-regular graph
    if(!strcmp("regular", type) || !strcmp("regular-const", type)) { 
        assert(argc-2 >= 7);
        n               = atol(argv[2+0]);
        d               = atol(argv[2+1]);
        max_t           = atol(argv[2+2]);
        max_rt          = atol(argv[2+3]);
        num_sources     = atol(argv[2+4]);
        num_separators  = atol(argv[2+5]);
        seed            = atol(argv[2+6]);

        // error check
        assert(n >= 1 && d >= 0 && n*d % 2 == 0 && seed >= 1);
        assert(max_t >= 1);
        assert(max_rt >= 1);
        assert(num_sources >= 1 && num_sources <= n);
        assert(num_separators >=0 && num_separators <= n);

        //srand(seed);

        // generate graph
        g = graph_reg_rand(n, d, max_t, max_rt, seed);
        g->gtype = REGULAR;
    }

    // powerlaw graph
    if(!strcmp("powlaw", type) || !strcmp("powlaw-const", type)) { 
        assert(argc-2 >= 9);
        n               = atol(argv[2+0]);
        d               = atol(argv[2+1]);
        al              = atof(argv[2+2]);
        w               = atol(argv[2+3]);
        max_t           = atol(argv[2+4]);
        max_rt          = atol(argv[2+5]);
        num_sources     = atol(argv[2+6]);
        num_separators  = atol(argv[2+7]);
        seed            = atol(argv[2+8]);

        // error check
        assert(n >= 1 && d >= 0 && al < 0.0 && w >= 2 && w <= n && seed >= 1);
        assert(max_t >= 1);
        assert(max_rt >= 1);
        assert(num_sources >= 1 && num_sources <= n);
        assert(num_separators >=0 && num_separators <= n);

        // generate graph
        g = graph_powlaw_rand(n, d, max_t, max_rt, al, w, seed);
        g->gtype = POWLAW;
    }


    srand(seed);
    // set sources
    index_t *sources = (index_t *) 0;
    get_random_vertices(num_sources, n, irand(), &sources);

    // set separators
    index_t *separators = (index_t *) 0;
    get_random_vertices(num_separators, n, irand(), &separators);

    // set resting time
    index_t *rest_time = (index_t *) 0;
    if(!strcmp("regular-const", type) || !strcmp("powlaw-const", type)) {
        get_rest_time_const(n, max_rt, &rest_time);
    } else {
        get_rest_time_random(n, max_rt, irand(), &rest_time);
    }

    graph_out_dimacs(stdout, g, max_rt);
    graph_out_rest_time(stdout, n, rest_time);
    graph_out_sources(stdout, num_sources, sources);
    graph_out_separators(stdout, num_separators, separators);

    FREE("vertices", sources);
    FREE("vertices", separators);
    FREE("rest_time", rest_time);
    graph_free(g);

        /*
    // powerlaw graph
    if(!strcmp("powlaw", type) || !strcmp("powlaw-const", type)) { 
        assert(argc-2 >= 9);
        index_t n           = atol(argv[2+0]);
        index_t d           = atol(argv[2+1]);
        double al           = atof(argv[2+2]);
        index_t w           = atol(argv[2+3]);
        index_t k           = atol(argv[2+4]);
        index_t max_t       = atol(argv[2+5]);
        index_t max_rt      = atol(argv[2+6]);
        index_t num_sources = atol(argv[2+7]);
        index_t seed        = atol(argv[2+8]);

        // error check
        assert(n >= 1 && d >= 0 && al < 0.0 && w >= 2 && w <= n && k >= 1 && k <= n && seed >= 1);
        assert(max_t >= 1);
        assert(max_rt >= 1);
        assert(num_sources >= 1 && num_sources <= n);

        // generate graph
        g = graph_powlaw_rand(n, d, max_t, max_rt, al, w, seed);
        g->gtype = POWLAW;

        // set sources
        index_t *sources = (index_t *) MALLOC("sources", sizeof(index_t)*num_sources);
        for(index_t i = 0; i < num_sources; i++)
            sources[i] = UNDEFINED;
        graph_set_rand_sources(k, num_sources, sources, g, seed);

        if(!strcmp("powlaw-const", type)) {
            graph_set_rest_time(n, max_rt, REST_TIME_CONST, g, seed);
        }

        graph_out_dimacs(stdout, g);
        graph_out_sources(stdout, num_sources, sources);

        FREE("sources", sources);
        graph_free(g);
    }
    */

    /*
    // d-regular graphs (for testing)
    if(!strcmp("regular-test", type)) {
        assert(argc-2 >= 7);
        index_t n           = atol(argv[2+0]);
        index_t d           = atol(argv[2+1]);
        index_t k           = atol(argv[2+2]);
        index_t max_t       = atol(argv[2+3]);
        index_t max_rt      = atol(argv[2+4]);
        index_t num_targets = atol(argv[2+5]);
        index_t seed        = atol(argv[2+6]);

        index_t max_c       = 1; //init max colors

        assert(n >= 1 && d >= 0 && n*d % 2 == 0 && k >= 1 && k <= n && seed >= 1);
        assert(max_t >= 1 && max_c >= 1 && max_c <= 32);
        assert(max_rt >= 1);
        assert(num_targets >= 1 && num_targets <= n/k);

        g = graph_reg_rand_test(n, d, max_t, max_rt, num_targets, seed);
        g->gtype = REGULAR;
        if(!strcmp(argv[argc-1],"-dir") || !strcmp(argv[argc-2],"-dir"))
            g->is_directed = 1;

        index_t num_sources = 1;
        index_t *sources    = (index_t *) MALLOC("sources", sizeof(index_t)*num_sources);
        index_t *targets    = (index_t *) MALLOC("targets", sizeof(index_t)*num_targets);

        for(index_t i = 0; i < num_sources; i++)
            sources[i] = UNDEFINED;
        for(index_t i = 0; i < num_targets; i++)
            targets[i] = UNDEFINED;

        graph_set_rand_target_test(g, sources, targets, k, max_c, max_t, num_sources, num_targets, seed);
       
        graph_out_dimacs(stdout, g);

        graph_out_sources(stdout, num_sources, sources);
        graph_out_targets(stdout, num_targets, targets);

        FREE("sources", sources);
        FREE("targets", targets);
        graph_free(g);
    }
    */

    fprintf(stderr, "gen-count [%s, %s]: n = %ld, m = %ld, t = %ld, rt = %ld\n", 
            type,
            g->is_directed ? "directed" : "undirected",
            g->num_vertices,
            g->num_edges,
            g->time_limit,
            max_rt
            );

    assert(malloc_balance == 0);
    return 0;
}
