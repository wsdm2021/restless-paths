/*
 * `Restless reachability in temporal graphs: algebraic methods and applications`
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
 * Copyright (c) 2020 Anonymous authors
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
#include<ctype.h>
#include<omp.h>

/************************************************************* Configuration. */
#define MAX_K          32
#define MAX_SHADES     32

#define PREFETCH_PAD   32
#define MAX_THREADS   128

#define UNDEFINED -1
#define MATH_INF ((index_t)0x3FFFFFFF)

#include"builds.h"        // get build config

typedef long int index_t; // default to 64-bit indexing

#include"gf.h"       // finite fields
#include"ffprng.h"   // fast-forward pseudorandom number generator


#define MIN(x,y) (x)<(y) ? (x) : (y)
#define MAX(x,y) (x)>(y) ? (x) : (y)

/********************************************************************* Flags. */


/************************************************************* Common macros. */

/* Linked list navigation macros. */

#define pnlinknext(to,el) { (el)->next = (to)->next; (el)->prev = (to); (to)->next->prev = (el); (to)->next = (el); }
#define pnlinkprev(to,el) { (el)->prev = (to)->prev; (el)->next = (to); (to)->prev->next = (el); (to)->prev = (el); }
#define pnunlink(el) { (el)->next->prev = (el)->prev; (el)->prev->next = (el)->next; }
#define pnrelink(el) { (el)->next->prev = (el); (el)->prev->next = (el); }


/*********************************************************** Error reporting. */

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

/********************************************************* Get the host name. */

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

/********************************************************* Available threads. */

index_t num_threads(void)
{
#ifdef BUILD_PARALLEL
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/********************************************** Memory allocation & tracking. */

#define MALLOC(x) malloc_wrapper(x)
#define FREE(x) free_wrapper(x)

index_t malloc_balance = 0;

struct malloc_track_struct
{
    void *p;
    size_t size;
    struct malloc_track_struct *prev;
    struct malloc_track_struct *next;
};

typedef struct malloc_track_struct malloc_track_t;

malloc_track_t malloc_track_root;
size_t malloc_total = 0;

#define MEMTRACK_STACK_CAPACITY 256
size_t memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t memtrack_stack_top = -1;

void *malloc_wrapper(size_t size)
{
    if(malloc_balance == 0) {
        malloc_track_root.prev = &malloc_track_root;
        malloc_track_root.next = &malloc_track_root;
    }
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;

    malloc_track_t *t = (malloc_track_t *) malloc(sizeof(malloc_track_t));
    t->p = p;
    t->size = size;
    pnlinkprev(&malloc_track_root, t);
    malloc_total += size;
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < malloc_total)
            memtrack_stack[i] = malloc_total;    
    return p;
}

void free_wrapper(void *p)
{
    malloc_track_t *t = malloc_track_root.next;
    for(;
        t != &malloc_track_root;
        t = t->next) {
        if(t->p == p)
            break;
    }
    if(t == &malloc_track_root)
        ERROR("FREE issued on a non-tracked pointer %p", p);
    malloc_total -= t->size;
    pnunlink(t);
    free(t);
    
    free(p);
    malloc_balance--;
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC(sizeof(index_t)*n);
    return t;
}

void push_memtrack(void) 
{
    assert(memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    memtrack_stack[++memtrack_stack_top] = malloc_total;
}

size_t pop_memtrack(void)
{
    assert(memtrack_stack_top >= 0);
    return memtrack_stack[memtrack_stack_top--];    
}

size_t current_mem(void)
{
    return malloc_total;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}

void print_current_mem(void)
{
    fprintf(stdout, "{curr: %.2lfGiB}", inGiB(current_mem()));
    fflush(stdout);
}

void print_pop_memtrack(void)
{
    fprintf(stdout, "{peak: %.2lfGiB}", inGiB(pop_memtrack()));
    fflush(stdout);
}

/******************************************************** Timing subroutines. */

#define TIME_STACK_CAPACITY 256
double start_stack[TIME_STACK_CAPACITY];
index_t start_stack_top = -1;

void push_time(void) 
{
    assert(start_stack_top + 1 < TIME_STACK_CAPACITY);
    start_stack[++start_stack_top] = omp_get_wtime();
}

double pop_time(void)
{
    double wstop = omp_get_wtime();
    assert(start_stack_top >= 0);
    double wstart = start_stack[start_stack_top--];
    return (double) (1000.0*(wstop-wstart));
}

/******************************************************************* Sorting. */

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

#define LEFT(x)      (x<<1)
#define RIGHT(x)     ((x<<1)+1)
#define PARENT(x)    (x>>1)

void heapsort_indext(index_t n, index_t *a)
{
    /* Shift index origin from 0 to 1 for convenience. */
    a--; 
    /* Build heap */
    for(index_t i = 2; i <= n; i++) {
        index_t x = i;
        while(x > 1) {
            index_t y = PARENT(x);
            if(a[x] <= a[y]) {
                /* heap property ok */
                break;              
            }
            /* Exchange a[x] and a[y] to enforce heap property */
            index_t t = a[x];
            a[x] = a[y];
            a[y] = t;
            x = y;
        }
    }

    /* Repeat delete max and insert */
    for(index_t i = n; i > 1; i--) {
        index_t t = a[i];
        /* Delete max */
        a[i] = a[1];
        /* Insert t */
        index_t x = 1;
        index_t y, z;
        while((y = LEFT(x)) < i) {
            z = RIGHT(x);
            if(z < i && a[y] < a[z]) {
                index_t s = z;
                z = y;
                y = s;
            }
            /* Invariant: a[y] >= a[z] */
            if(t >= a[y]) {
                /* ok to insert here without violating heap property */
                break;
            }
            /* Move a[y] up the heap */
            a[x] = a[y];
            x = y;
        }
        /* Insert here */
        a[x] = t; 
    }
}

/*************************************************** Random numbers and such. */

index_t irand(void)
{
    return (((index_t) rand())<<31)^((index_t) rand());
}

/***************************************************** (Parallel) prefix sum. */

index_t prefixsum(index_t n, index_t *a, index_t k)
{

#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = (stop-start+1)*k;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    index_t run = 0;
    for(index_t t = 1; t <= nt; t++) {
        index_t v = s[t-1];
        s[t-1] = run;
        run += v;
    }
    s[nt] = run;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t trun = s[t];
        for(index_t u = start; u <= stop; u++) {
            index_t tv = a[u];
            a[u] = trun;
            trun += tv + k;
        }
        assert(trun == s[t+1]);    
    }

#else

    index_t run = 0;
    for(index_t u = 0; u < n; u++) {
        index_t tv = a[u];
        a[u] = run;
        run += tv + k;
    }

#endif

    return run; 
}

/************************************************************* Parallel sum. */

index_t parallelsum(index_t n, index_t *a)
{
    index_t sum = 0;
#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = 0;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    for(index_t t = 0; t < nt; t++)
        sum += s[t];
#else
    for(index_t i = 0; i < n; i++) {
        sum += a[i];
    }
#endif
    return sum;
}

// count number of non-zero values in an array
index_t parallelcount(index_t n, index_t *a)
{
    index_t total_cnt = 0;
#ifdef BUILD_PARALLEL
    index_t nt = num_threads();
    index_t block_size = n/nt;
    index_t *cnt_nt = alloc_idxtab(nt);
#pragma omp parallel for
    for(index_t th = 0; th <nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        index_t cnt = 0;
        for(index_t i = start; i <= stop; i++)
            cnt += (a[i] ? 1 : 0);

        cnt_nt[th] = cnt;
    }

    for(index_t th = 0; th < nt; th++)
        total_cnt += cnt_nt[th];  
#else
    for(index_t i = 0; i < n; i++)
        total_cnt += (a[i] ? 1 : 0);
#endif
    return total_cnt;
}


/************************ Search for an interval of values in a sorted array. */

index_t get_interval(index_t n, index_t *a, 
                            index_t lo_val, index_t hi_val,
                            index_t *iv_start, index_t *iv_end)
{
    assert(n >= 0);
    if(n == 0) {
        *iv_start = 0; 
        return 0;
    }
    assert(lo_val <= hi_val);
    // find first element in interval (if any) with binary search
    index_t lo = 0;
    index_t hi = n-1;
    // at or above lo, and at or below hi (if any)
    while(lo < hi) {
        index_t mid = (lo+hi)/2; // lo <= mid < hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi (if any)
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo (if any), lo <= hi
            else
                hi = mid;   // at or below hi (exists) 
        }
        // 0 <= lo <= n-1
    }
    if(a[lo] < lo_val || a[lo] > hi_val) {
        // array contains no values in interval
        if(a[lo] < lo_val) {
            lo++;
            assert(lo == n || a[lo+1] > hi_val);
        } else {
            assert(lo == 0 || a[lo-1] < lo_val);
        }
        *iv_start = lo; 
        *iv_end   = hi;
        return 0; 
    }
    assert(lo_val <= a[lo] && a[lo] <= hi_val);
    *iv_start = lo;
    // find interval end (last index in interval) with binary search
    lo = 0;
    hi = n-1;
    // last index (if any) is at or above lo, and at or below hi
    while(lo < hi) {
        index_t mid = (lo+hi+1)/2; // lo < mid <= hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi, lo <= hi
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo
            else
                lo = mid;   // at or above lo, lo <= hi
        }
    }
    assert(lo == hi);
    *iv_end = lo; // lo == hi
    return 1+*iv_end-*iv_start; // return cut size
}


/******************************************************************** Stack. */

typedef struct stack_node {
    index_t u;
    index_t l;
    index_t t;
} stack_node_t;

typedef struct stack {
    index_t size; // size of stack
    index_t n; // number of elements
    stack_node_t *a;
}stk_t;

stk_t * stack_alloc(index_t size)
{
    stk_t *s = (stk_t *) malloc(sizeof(stk_t)); 
    s->size = size;
    s->n = 0;
    s->a = (stack_node_t *) malloc(s->size*sizeof(stack_node_t));

#ifdef DEBUG
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        e->u = UNDEFINED;
        e->l = UNDEFINED;
        e->t = UNDEFINED;
    }
#endif
    return s;
}

void stack_free(stk_t *s)
{
    free(s->a);
    free(s);
}

void stack_push(stk_t *s, stack_node_t *e_in)
{
    assert(s->n < s->size);
    stack_node_t *e = s->a + s->n;
    e->u = e_in->u;
    e->l = e_in->l;
    e->t = e_in->t;
    s->n++;
}

void stack_pop(stk_t *s, stack_node_t *e_out)
{
    assert(s->n > 0);
    s->n--;
    stack_node_t *e = s->a + s->n;
    e_out->u = e->u;
    e_out->l = e->l;
    e_out->t = e->t;

#ifdef DEBUG
    e->u = UNDEFINED;
    e->l = UNDEFINED;
    e->t = UNDEFINED;
#endif
}

void stack_top(stk_t *s, stack_node_t *e_out)
{
    assert(s->n >= 0);
    stack_node_t *e = s->a + s->n-1;
    e_out->u = e->u;
    e_out->l = e->l;
    e_out->t = e->t;
}

void stack_empty(stk_t *s)
{
    s->n = 0;
}

void stack_get_vertices(stk_t *s, index_t *uu)
{
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        uu[i] = e->u;
    }
}

void stack_get_timestamps(stk_t *s, index_t *tt)
{
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        tt[i] = e->t;
    }
}

#ifdef DEBUG
void print_stack(stk_t *s)
{
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "print stack\n");
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "size: %ld\n", s->size);
    fprintf(stdout, "n: %ld\n", s->n);
    fprintf(stdout, "a: ");
    for(index_t i = 0; i < s->n; i++) {
        stack_node_t *e = s->a + i;
        fprintf(stdout, "[%ld, %ld, %ld]%s", e->u, e->l, e->t, (i==s->n-1)?"\n":" ");
    }
    fprintf(stdout, "-----------------------------------------------\n");
}

void print_stacknode(stack_node_t *e)
{
    fprintf(stdout, "print stack-node: [%ld, %ld, %ld]\n", e->u, e->l, e->t);
}
#endif

/****************************************************************** Sieving. */

long long int num_muls;
long long int trans_bytes;

#define SHADE_LINES ((MAX_SHADES+SCALARS_IN_LINE-1)/SCALARS_IN_LINE)
typedef unsigned int shade_map_t;

void constrained_sieve_pre(index_t         n,
                           index_t         k,
                           index_t         g,
                           index_t         pfx,
                           index_t         num_shades,
                           shade_map_t     *d_s,
                           ffprng_scalar_t seed,
                           line_array_t    *d_x)
{
    assert(g == SCALARS_IN_LINE);   
    assert(num_shades <= MAX_SHADES);

    line_t   wdj[SHADE_LINES*MAX_K];

    ffprng_t base;
    FFPRNG_INIT(base, seed);
    for(index_t j = 0; j < k; j++) {
        for(index_t dl = 0; dl < SHADE_LINES; dl++) {
            index_t jsdl = j*SHADE_LINES+dl;
            LINE_SET_ZERO(wdj[jsdl]);
            for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                ffprng_scalar_t rnd;
                FFPRNG_RAND(rnd, base);
                scalar_t rs = (scalar_t) rnd;
                LINE_STORE_SCALAR(wdj[jsdl], a, rs);   // W: [cached]
            }
        }
    }

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        FFPRNG_FWD(gen, SHADE_LINES*SCALARS_IN_LINE*start, base);
        line_t vd[SHADE_LINES];
        for(index_t j = 0; j < SHADE_LINES; j++) {
            LINE_SET_ZERO(vd[j]); // to cure an annoying compiler warning
        }       
        for(index_t u = start; u <= stop; u++) {
            scalar_t uu[MAX_K];
            shade_map_t shades_u = d_s[u];            // R: n   shade_map_t
            for(index_t dl = 0; dl < SHADE_LINES; dl++) {
                for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                    index_t d = dl*SCALARS_IN_LINE + a;
                    ffprng_scalar_t rnd;
                    FFPRNG_RAND(rnd, gen);
                    scalar_t rs = (scalar_t) rnd;
                    rs = rs & (-((scalar_t)((shades_u >> d)&(d < num_shades))));  
                    LINE_STORE_SCALAR(vd[dl], a, rs); // W: [cached]
                }
            }
            for(index_t j = 0; j < k; j++) {
                scalar_t uj;
                SCALAR_SET_ZERO(uj);
                for(index_t dl = 0; dl < SHADE_LINES; dl++) {
                    index_t jsdl = j*SHADE_LINES+dl;
                    line_t ln;
                    LINE_MUL(ln, wdj[jsdl], vd[dl]);  // R: [cached]
                                                      // MUL: n*SHADE_LINES*g*k
                    scalar_t lns;
                    LINE_SUM(lns, ln);
                    SCALAR_ADD(uj, uj, lns);
                }
                uu[j] = uj;
            }
            line_t ln;
            LINE_SET_ZERO(ln);
            for(index_t a = 0; a < SCALARS_IN_LINE; a++) {
                index_t ap = a < (1L << k) ? pfx+a : 0;
                scalar_t xua;
                SCALAR_SET_ZERO(xua);
                for(index_t j = 0; j < k; j++) {
                    scalar_t z_uj = uu[j];            // R: [cached]
                    z_uj = z_uj & (-((scalar_t)(((ap) >> j)&1)));
                    SCALAR_ADD(xua, xua, z_uj);
                }
                LINE_STORE_SCALAR(ln, a, xua);
            }
            LINE_STORE(d_x, u, ln);                  // W: ng scalar_t
        }
    }

    num_muls    += n*SHADE_LINES*g*k;
    trans_bytes += sizeof(scalar_t)*n*g + sizeof(shade_map_t)*n;
}

/***************************************************************** Line sum. */

scalar_t line_sum(index_t      l, 
                  index_t      g,
                  line_array_t *d_s)
{

    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    scalar_t ts[MAX_THREADS];
    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        SCALAR_SET_ZERO(ts[t]);
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        line_t acc;
        LINE_SET_ZERO(acc);
        for(index_t i = start; i <= stop; i++) {    
            LINE_LOAD(ln, d_s, i);    // R: lg scalar_t
            LINE_ADD(acc, acc, ln);
        }
        scalar_t lsum;
        LINE_SUM(lsum, acc);
        ts[t] = lsum;
    }
    scalar_t sum;
    SCALAR_SET_ZERO(sum);
    for(index_t t = 0; t < nt; t++) {
        SCALAR_ADD(sum, sum, ts[t]);
    }

    trans_bytes += sizeof(scalar_t)*l*g;
    return sum;
}

void vertex_acc(index_t      l, // n
                index_t      g, // g
                index_t      stride, // k
                line_array_t *d_s,
                scalar_t     *out)
{
    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    //scalar_t ts[MAX_THREADS];

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        //SCALAR_SET_ZERO(ts[t]);
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        scalar_t lsum;
        for(index_t i = start; i <= stop; i++) {
            LINE_LOAD(ln, d_s, i);    // R: lg scalar_t
            LINE_SUM(lsum, ln);
            out[i] ^= lsum;            // R: scalar_t,  W: scalar_t
        }
    }
    //scalar_t sum;
    //SCALAR_SET_ZERO(sum);
    //for(index_t t = 0; t < nt; t++) {
    //    SCALAR_ADD(sum, sum, ts[t]);
    //}

    trans_bytes += sizeof(scalar_t)*(l*g+2);
}




scalar_t line_sum_stride(index_t      l, 
                         index_t      g,
                         index_t      stride,
                         line_array_t *d_s)
{

    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    scalar_t ts[MAX_THREADS];
    
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        SCALAR_SET_ZERO(ts[th]);
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        line_t acc;
        LINE_SET_ZERO(acc);
        for(index_t i = start; i <= stop; i++) {    
            index_t ii = i*stride;
            LINE_LOAD(ln, d_s, ii);    // R: lg scalar_t
            LINE_ADD(acc, acc, ln);
        }
        scalar_t lsum;
        LINE_SUM(lsum, acc);
        ts[th] = lsum;
    }
    scalar_t sum;
    SCALAR_SET_ZERO(sum);
    for(index_t th = 0; th < nt; th++) {
        SCALAR_ADD(sum, sum, ts[th]);
    }

    trans_bytes += sizeof(scalar_t)*l*g;

    return sum;
}

void vertex_acc_stride(index_t      l,
                       index_t      g,
                       index_t      stride,
                       line_array_t *d_s,
                       scalar_t     *out)
{
    index_t nt = num_threads();
    index_t block_size = l/nt;
    assert(nt < MAX_THREADS);
    //scalar_t ts[MAX_THREADS];

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        //SCALAR_SET_ZERO(ts[th]);
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? l-1 : (start+block_size-1);
        line_t ln;
        scalar_t lsum;
        for(index_t i = start; i <= stop; i++) {
            index_t ii = i*stride;
            LINE_LOAD(ln, d_s, ii);    // R: lg scalar_t
            LINE_SUM(lsum, ln);
            out[i] ^= lsum;            // R: scalar_t,  W: scalar_t
        }
    }
    //scalar_t sum;
    //SCALAR_SET_ZERO(sum);
    //for(index_t th = 0; th < nt; th++) {
    //    SCALAR_ADD(sum, sum, ts[th]);
    //}

    trans_bytes += sizeof(scalar_t)*(l*g+2);
}


/***************************************** k-temppath generating function. */
#ifdef DEBUG

#define PRINT_LINE(source)                                                  \
{                                                                           \
    scalar_t *s = (scalar_t *)&source;                                      \
    for(index_t i = 0; i < SCALARS_IN_LINE; i++) {                          \
        fprintf(stdout, SCALAR_FORMAT_STRING"%s",                           \
                        (long) s[i],                                        \
                        i==SCALARS_IN_LINE-1 ? "\n":" ");                   \
    }                                                                       \
}
#endif


#if BUILD_GENF == 2

#define TEMP_PATH_LINE_IDX2(n, k, tmax, u, l, i) (((u)*(tmax+1))+(i))

#ifdef DEBUG
void print_ds(index_t n,
                index_t tmax,
                line_array_t *d_s)
{
    for(index_t u = 0; u < n; u++) {
        fprintf(stdout, "--------------------------------------------------\n");
        fprintf(stdout, "u: %ld\n", u+1);
        fprintf(stdout, "--------------------------------------------------\n");
        for(index_t i = 0; i <= tmax; i++) {
            fprintf(stdout, "%ld: ", i);
            index_t i_uli = TEMP_PATH_LINE_IDX2(n, k, tmax, 1, i, u);
            line_t p_uli;
            LINE_LOAD(p_uli, d_s, i_uli);
            PRINT_LINE(p_uli);
            scalar_t sum;
            LINE_SUM(sum, p_uli);
            fprintf(stdout, "line sum: "SCALAR_FORMAT_STRING"\n",sum);
        }
    }
}
#endif

scalar_t vloc_finegrain(index_t       n,
                        index_t       g,
                        index_t       k,
                        index_t       tmax,
                        line_array_t  *d_s,
                        scalar_t      *out)
{
    index_t nt = num_threads();
    index_t block_size = n/nt;
    assert(nt < MAX_THREADS);
    scalar_t tsum[MAX_THREADS];

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        SCALAR_SET_ZERO(tsum[th]);
        index_t  start = th*block_size;
        index_t  stop = (th == nt-1) ? n-1 : (start+block_size-1);
        line_t   ln;
        scalar_t lsum;
        scalar_t acc;
        SCALAR_SET_ZERO(acc);
        for(index_t u = start; u <= stop; u++) {
            index_t i_ul0 = TEMP_PATH_LINE_IDX2(n, k, tmax, u, k, 0);
            index_t i_u0  = (u*(tmax+1));
            for(index_t i = 0; i <= tmax; i++) {
                index_t i_uli = i_ul0 + i;
                index_t i_ui  = i_u0 + i;
                LINE_LOAD(ln, d_s, i_uli);
                LINE_SUM(lsum, ln);
                out[i_ui] ^= lsum;
                acc ^= lsum;
            }
        }
        tsum[th] = acc;
    }

    scalar_t sum;
    SCALAR_SET_ZERO(sum);
    for(index_t th = 0; th < nt; th++)
        SCALAR_ADD(sum, sum, tsum[th]);

    //TODO: update bandwidth computation
    trans_bytes += LINE_ARRAY_SIZE((tmax+1)*n*g);

    return sum;
}


void init_ds(index_t n,
             index_t k,
             index_t tmax,
             line_array_t *d_s)
{
    line_t p_zero;
    LINE_SET_ZERO(p_zero);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t i_u10 = TEMP_PATH_LINE_IDX2(n, k, tmax, u, 1, 0);
        for(index_t i= 0; i <= tmax; i++) {
            index_t i_u1i = i_u10 + i;
            LINE_STORE(d_s, i_u1i, p_zero);
        }
    }
}


void k_temp_path_round(index_t n,
                       index_t m,
                       index_t k,
                       index_t tmax,
                       index_t rtmax,
                       index_t t,
                       index_t g,
                       index_t l,
                       index_t *d_pos,
                       index_t *d_adj,
                       index_t yl_seed,
                       index_t *rtime,
                       line_array_t *d_x,
                       line_array_t *d_l1,
                       line_array_t *d_l)
{
    assert(g == SCALARS_IN_LINE);

    index_t nt = num_threads();
    index_t length = n;
    index_t block_size = length/nt;

    index_t i = t;
    ffprng_t y_base;
    FFPRNG_INIT(y_base, yl_seed);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? length-1 : (start+block_size-1);
        ffprng_t y_gen;
        // forward the psuedo-random number generator
        index_t y_pos = (d_pos[start] - start) * rtmax;
        FFPRNG_FWD(y_gen, y_pos, y_base);
        for(index_t u = start; u <= stop; u++) {
            index_t pu  = d_pos[n*(i-1)+u];             
            index_t deg = d_adj[pu];               
            line_t p_uli;
            LINE_SET_ZERO(p_uli);
            for(index_t d = 1; d <= deg; d++) {
                index_t v = d_adj[pu+d];          
                index_t dv = (i > rtime[v]) ? rtime[v] : i-1; // i-j > 0
                index_t i_vl1idv = TEMP_PATH_LINE_IDX2(n, k, tmax, v, l-1, i-dv);
                for(index_t j = 0; j <= dv; j++) { 
                    line_t p_vl1ij;
                    index_t i_vl1ij = i_vl1idv + j;
                    LINE_LOAD(p_vl1ij, d_l1, i_vl1ij);
#ifdef BUILD_PREFETCH
                    // prefetch next line P_{v,l-1,i-j+1}
                    index_t i_vl1ij1 = i_vl1idv + (j==dv) ? dv : j+1;
                    LINE_PREFETCH(d_l1, i_vl1ij1);
#endif
                    ffprng_scalar_t rnd;
                    FFPRNG_RAND(rnd, y_gen);
                    scalar_t y_uvlij = (scalar_t) rnd;
                    line_t sy;
                    LINE_MUL_SCALAR(sy, p_vl1ij, y_uvlij);
                    LINE_ADD(p_uli, p_uli, sy);
                }
            }
            line_t xu;
            LINE_LOAD(xu, d_x, u);
            LINE_MUL(p_uli, p_uli, xu);

            index_t i_uli = TEMP_PATH_LINE_IDX2(n, k, tmax, u, l, i);
            LINE_STORE(d_l, i_uli, p_uli);      // W: ng  scalar_t
        }
    }

    //TODO: update bandwidth computation
    // total edges at time `i`
    index_t m_i = d_pos[n*(i-1) + n-1] - d_pos[n*(i-1)] - (n-1) + 
                  d_adj[d_pos[n*(i-1)+(n-1)]];
    trans_bytes += ((2*n*tmax)+m_i)*sizeof(index_t) + (2*n+m_i)*g*sizeof(scalar_t);
    num_muls    += (n*g+m_i);
}


scalar_t k_temp_path(index_t         n,
                     index_t         m,
                     index_t         k,
                     index_t         tmax,
                     index_t         rtmax,
                     index_t         g,
                     index_t         vert_loc,
                     index_t         *d_pos,
                     index_t         *d_adj, 
                     ffprng_scalar_t y_seed,
                     index_t         *rtime,
                     line_array_t    *d_x,
                     scalar_t        *vsum) 
{
    assert( g == SCALARS_IN_LINE);
    assert( k >= 1);

    line_array_t *d_l1 = (line_array_t *) MALLOC(LINE_ARRAY_SIZE((tmax+1)*n*g));
    line_array_t *d_l  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE((tmax+1)*n*g));

    init_ds(n, 1, tmax, d_l);

    // initialise: l = 1
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t i_u10 = TEMP_PATH_LINE_IDX2(n, k, tmax, u, 1, 0);
        for(index_t i = 0; i <= tmax; i++) {
            line_t xu;
            LINE_LOAD(xu, d_x, u);
            index_t i_u1i = i_u10 + i;
            LINE_STORE(d_l1, i_u1i, xu);
        }
    }

    srand(y_seed);
    for(index_t l = 2; l <= k; l++) {
        for(index_t i = l-1; i <= tmax; i++) {
            ffprng_scalar_t yl_seed = irand(); // new seed for each l
            k_temp_path_round(n, m, k, tmax, rtmax, i, g, l, 
                              d_pos, d_adj, yl_seed, rtime, d_x, 
                              d_l1, d_l);
        }

        // swap and initialise
        line_array_t *d_temp = d_l1;
        d_l1 = d_l;
        d_l = d_temp;
        init_ds(n, 1, tmax, d_l);
    }

    // sum up
    //index_t ii = TEMP_PATH_LINE_IDX2(n, k, tmax, 1, tmax, 0);
    scalar_t sum = vloc_finegrain(n, g, k, tmax, d_l1, vsum);
    // free memory
    FREE(d_l1);
    FREE(d_l);

    return sum;
}
#endif

/************************************************************ The oracle(s). */

index_t temppath_oracle(index_t         n,
                        index_t         k,
                        index_t         tmax,
                        index_t         rtmax,
                        index_t         *h_pos,
                        index_t         *h_adj,
                        index_t         num_shades,
                        index_t         *rtime,
                        shade_map_t     *h_s,
                        ffprng_scalar_t y_seed,
                        ffprng_scalar_t z_seed,
                        index_t         vert_loc,
                        scalar_t        *master_vsum) 
{
    push_memtrack();
    assert(k >= 1 && k < 31);
    //index_t m = h_pos[n-1]+h_adj[h_pos[n-1]]+1-n;
    index_t m = h_pos[n*(tmax-1)+n-1]+h_adj[h_pos[n*(tmax-1)+n-1]]+1-(n*tmax);
    index_t sum_size = 1 << k;       

    index_t g = SCALARS_IN_LINE;
    index_t outer = (sum_size + g-1) / g; 
    // number of iterations for outer loop

    num_muls = 0;
    trans_bytes = 0;

    index_t *d_pos     = h_pos;
    index_t *d_adj     = h_adj;
    line_array_t *d_x  = (line_array_t *) MALLOC(LINE_ARRAY_SIZE(n*g));
    
    /* Run the work & time it. */
    push_time();

    scalar_t master_sum;
    SCALAR_SET_ZERO(master_sum);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t i_u0 = (u*(tmax+1));
        for(index_t i = 0; i <= tmax; i++) {
            index_t i_ui = i_u0 + i;
            SCALAR_SET_ZERO(master_vsum[i_ui]);
        }
    }
   

    for(index_t out = 0; out < outer; out++) {
        // Eq. (3)
        constrained_sieve_pre(n, k, g, g*out, num_shades, h_s, z_seed, d_x);
#define GENF_TYPE "restless_path_genf"
        // Eq. (4)
        scalar_t sum = k_temp_path(n, m, k, tmax, rtmax, g, vert_loc, 
                                   d_pos, d_adj, y_seed, rtime, d_x, 
                                   master_vsum);
        SCALAR_ADD(master_sum, master_sum, sum);
    }

    double time = pop_time();
    //double trans_rate = trans_bytes / (time/1000.0);
    //double mul_rate = num_muls / time;
    FREE(d_x);

    fprintf(stdout, 
            SCALAR_FORMAT_STRING
            " %.2lf ms"
            //" [%.2lfGiB/s, %.2lfGHz]"
            " %d",
            (long) master_sum,
            time,
            //trans_rate/((double) (1 << 30)),
            //mul_rate/((double) 1e6),
            master_sum != 0);
    fprintf(stdout, " ");
    print_pop_memtrack();
    fprintf(stdout, " ");   
    print_current_mem();   
    fflush(stdout);

    return master_sum != 0;
}

/************************************************* Rudimentary graph builder. */

typedef struct 
{
    index_t is_directed;
    index_t num_vertices;
    index_t num_edges;
    index_t max_time;
    index_t max_resttime;
    index_t edge_capacity;
    index_t *edges;
    index_t *rest_time;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC(sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE(was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);

    graph_t *g = (graph_t *) MALLOC(sizeof(graph_t));
    g->is_directed   = 0; // default: undirected graph
    g->num_vertices  = n;
    g->num_edges     = 0;
    g->edge_capacity = 100;
    g->edges  = enlarge(3*g->edge_capacity, 0, (void *) 0);

    g->rest_time = (index_t *) MALLOC(sizeof(index_t)*n);
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        g->rest_time[u] = UNDEFINED;

    return g;
}

void graph_free(graph_t *g)
{
    FREE(g->edges);
    FREE(g->rest_time);
    FREE(g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v, index_t t)
{
    assert(u >= 0 && 
           v >= 0 && 
           u < g->num_vertices &&
           v < g->num_vertices);
    assert(t>=0);
    //assert(t>=0 && t < g->max_time);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(6*g->edge_capacity, 3*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 3*g->num_edges;
    e[0] = u;
    e[1] = v;
    e[2] = t;
    g->num_edges++;
}

index_t *graph_edgebuf(graph_t *g, index_t cap)
{
    g->edges = enlarge(3*g->edge_capacity+3*cap, 3*g->edge_capacity, g->edges);
    index_t *e = g->edges + 3*g->num_edges;
    g->edge_capacity += cap;
    g->num_edges += cap;
    return e;
}

//void graph_set_color(graph_t *g, index_t u, index_t c)
//{
//    assert(u >= 0 && u < g->num_vertices && c >= 0);
//    g->colors[u] = c;
//}

void graph_set_is_directed(graph_t *g, index_t is_dir)
{
    assert(is_dir == 0 || is_dir == 1);
    g->is_directed = is_dir;
}

void graph_set_max_time(graph_t *g, index_t tmax)
{
    assert(tmax > 0);
    g->max_time = tmax;
}

void graph_set_resttime(graph_t *g, index_t u, index_t rt)
{
    assert(u >= 0 && u < g->num_vertices && rt >= 0 && rt <= g->max_resttime);
    g->rest_time[u] = rt;
}


void graph_set_max_resttime(graph_t *g, index_t rtmax)
{
    assert(rtmax > 0);
    g->max_resttime = rtmax;
}

#ifdef DEBUG
void print_graph(graph_t *g)
{
    index_t n = g->num_vertices;
    index_t m = g->num_edges;
    index_t tmax = g->max_time;
    index_t rtmax = g->max_resttime;
    index_t is_dir = g->is_directed;
    fprintf(stdout, "p motif %ld %ld %ld %ld %ld\n", n, m, tmax, rtmax, is_dir);

    index_t *e = g->edges;
    for(index_t i = 0; i < 3*m; i+=3) {
        fprintf(stdout, "e %ld %ld %ld\n", 
                        e[i]+1, e[i+1]+1, e[i+2]+1);
    }

    index_t *c = g->colors;
    for(index_t i = 0; i < n; i++)
        fprintf(stdout, "n %ld %ld\n", i+1, c[i]==UNDEFINED ? c[i] : c[i]+1);

    index_t *rt = g->rest_time;
    for(index_t i = 0; i < n; i++)
        fprintf(stdout, "r %ld %ld\n", i+1, rt[i]);
}
#endif


/************************************* Basic motif query processing routines. */

struct temppathq_struct
{
    index_t     is_stub;
    index_t     n;
    index_t     k;
    index_t     tmax;
    index_t     *pos;
    index_t     *adj;
    index_t     nl;
    index_t     *l;  
    index_t     ns;
    shade_map_t *shade;
    index_t     rtmax;
    index_t     *rtime;
    index_t     vert_loc;
    scalar_t    *vsum;
};

typedef struct temppathq_struct temppathq_t;

void adjsort(index_t n, index_t *pos, index_t *adj)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        heapsort_indext(deg, adj + pu + 1);
    }
}

void temppathq_free(temppathq_t *q)
{
    if(!q->is_stub) {
        FREE(q->pos);
        FREE(q->adj);
        FREE(q->l);
        FREE(q->shade);
        FREE(q->rtime);
        FREE(q->vsum);
    }
    FREE(q);
}

index_t temppathq_execute(temppathq_t *q)
{
    if(q->is_stub)
        return 0;
    return temppath_oracle(q->n, q->k, q->tmax, q->rtmax, q->pos, q->adj, q->ns, q->rtime, q->shade, 
                           irand(), irand(), q->vert_loc, q->vsum);
}

#ifdef DEBUG
void print_temppathq(temppathq_t *q)
{
    index_t n       = q->n;
    index_t k       = q->k;
    index_t tmax    = q->tmax;
    index_t *pos    = q->pos;
    index_t *adj    = q->adj;
    fprintf(stdout, "-----------------------------------------------\n");
    fprintf(stdout, "printing temppathq\n");
    fprintf(stdout, "is_stub = %ld\n", q->is_stub);
    fprintf(stdout, "n = %ld\n", n);
    fprintf(stdout, "k = %ld\n", k);
    fprintf(stdout, "tmax = %ld\n", tmax);
    fprintf(stdout, "pos\n");
    fprintf(stdout, "----\n ");
    for(index_t i = 0; i < n*tmax; i++) {
        fprintf(stdout, "%ld%s", pos[i], i%n==n-1 ? "\n ":" ");
    }

    fprintf(stdout, "adjacency list:\n");
    fprintf(stdout, "---------------\n");
    for(index_t t = 0; t < tmax; t++) {
        fprintf(stdout, "t: %ld\n", t+1);
        fprintf(stdout, "---------------\n");

        index_t *pos_t = pos + n*t;
        for(index_t u = 0; u < n; u++) {
            index_t pu = pos_t[u];
            index_t nu = adj[pu];
            index_t *adj_u = adj + pu + 1;
            fprintf(stdout, "%4ld:", u+1);
            for(index_t i = 0; i < nu; i++) {
                fprintf(stdout, " %4ld", adj_u[i]+1);
            }
            fprintf(stdout, "\n");
        }
    }

    index_t nl          = q->nl;
    index_t *l          = q->l;
    fprintf(stdout, "nl = %ld\n", nl);
    fprintf(stdout, "l:\n");
    for(index_t i = 0; i < nl; i++)
        fprintf(stdout, "%8ld : %8ld\n", nl, l[i]);

    index_t ns = q ->ns;
    shade_map_t *shade  = q->shade;
    fprintf(stdout, "ns : %ld\n", ns);
    fprintf(stdout, "shades:\n");
    for(index_t u = 0; u < n; u++)
        fprintf(stdout, "%10ld : 0x%08X\n", u+1, shade[u]);

    index_t rtmax = q->rtmax;
    index_t *rtime = q->rtime;
    fprintf(stdout, "rtmax: %ld", rtmax);
    fprintf(stdout, "rest time:\n");
    for(index_t u = 0; u < n; u++)
        fprintf(stdout, "%10ld : %8ld\n", u+1, rtime[u]);

    scalar_t *vsum = q->vsum;
    fprintf(stdout, "vert_loc: %ld\n", q->vert_loc);
    fprintf(stdout, "vsum:\n");
    for(index_t u = 0; u < n; u++) 
        fprintf(stdout, "%10ld : "SCALAR_FORMAT_STRING"\n", u+1, vsum[u]);
    fprintf(stdout, "-----------------------------------------------\n");
}

void print_array(const char *name, index_t n, index_t *a, index_t offset)
{
    fprintf(stdout, "%s (%ld):", name, n);
    for(index_t i = 0; i < n; i++) {
        fprintf(stdout, " %ld", a[i] == -1 ? -1 : a[i]+offset);
    }
    fprintf(stdout, "\n"); 
}
#endif

/******************************************************** Root query builder. */
// Query builder for directed graphs
temppathq_t *build_temppathq_dir(graph_t *g)
{
    push_memtrack();

    index_t n           = g->num_vertices;
    index_t m           = g->num_edges;
    index_t tmax        = g->max_time;
    index_t *pos        = alloc_idxtab(n*tmax);
    index_t *adj        = alloc_idxtab(n*tmax+2*m);
    index_t *rtime      = (index_t *) MALLOC(sizeof(index_t)*n);
    //index_t ns          = k;
    shade_map_t *shade  = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);

    temppathq_t *root = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    root->is_stub  = 0;
    root->n        = g->num_vertices;
    //root->k        = k;
    root->tmax     = tmax;
    root->pos      = pos;
    root->adj      = adj;
    root->nl       = 0;
    root->l        = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    //root->ns       = ns;
    root->shade    = shade;
    root->rtime    = rtime;
    root->vert_loc = 1;
    root->vsum     = (scalar_t *) MALLOC(sizeof(scalar_t)*(root->n)*(root->tmax+1));

    //assert(tmax >= k-1);

    push_time();
    fprintf(stdout, "build query: ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            //index_t u = e[j];
            index_t v = e[j+1];
            index_t t = e[j+2];
            index_t *pos_t = (pos + (n*t));
            //if(start <= u && u <= stop) {
            //    // I am responsible for u, record adjacency to u
            //    pos_t[u]++;
            //}
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                pos_t[v]++;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        //index_t u = e[j];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t *pos_t = pos + n*t;
        //pos_t[u]++;
        pos_t[v]++;
    }
#endif

    index_t run = prefixsum(n*tmax, pos, 1);
    assert(run == (n*tmax+m));
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
            adj[pos[u]] = 0;

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j+0];
            index_t v = e[j+1];
            index_t t = e[j+2];
            //if(start <= u && u <= stop) {
            //    // I am responsible for u, record adjacency to u
            //    index_t pu = pos[n*t+u];
            //    adj[pu + 1 + adj[pu]++] = v;
            //}
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                index_t pv = pos[n*t+v];
                adj[pv + 1 + adj[pv]++] = u;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j+0];
        index_t v = e[j+1];
        index_t t = e[j+2];
        //index_t pu = pos[n*t+u];
        index_t pv = pos[n*t+v];       
        //adj[pu + 1 + adj[pu]++] = v;
        adj[pv + 1 + adj[pv]++] = u;
    }
#endif

    time = pop_time();

    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    //print_temppathq(root);
    push_time();
    adjsort(n*tmax, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    // copy shades
    //push_time();
//#ifdef BUILD_PARALLEL
//#pragma omp parallel for
//#endif
    //for(index_t u = 0; u < n; u++) {
    //    shade_map_t s = 0;
    //    for(index_t j = 0; j < k; j++)
    //        if(colors[u] == kk[j])
    //            s |= 1UL << j;
    //    shade[u] = s;
    //    //fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
    //}
    //time = pop_time();
    //fprintf(stdout, "[shade: %.2lf ms] ", time);
    //fflush(stdout);

    // copy resting time
    push_time();
    index_t *rest_time = g->rest_time;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        rtime[u] = rest_time[u];
    time = pop_time();
    fprintf(stdout, "[rtime: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return root;
}

// Query builder for undirected graphs
//
temppathq_t *build_temppathq(graph_t *g)
{
    push_memtrack();

    index_t n           = g->num_vertices;
    index_t m           = g->num_edges;
    index_t tmax        = g->max_time;
    index_t *pos        = alloc_idxtab(n*tmax);
    index_t *adj        = alloc_idxtab(n*tmax+2*m);
    index_t *rtime      = (index_t *) MALLOC(sizeof(index_t)*n);
    //index_t ns          = k;
    shade_map_t *shade  = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);
     

    temppathq_t *root = (temppathq_t *) MALLOC(sizeof(temppathq_t));
    root->is_stub   = 0;
    root->n         = g->num_vertices;
    //root->k         = k;
    root->tmax      = tmax;
    root->pos       = pos;
    root->adj       = adj;
    root->nl        = 0;
    root->l         = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    //root->ns        = ns;
    root->shade     = shade;
    root->rtime     = rtime;
    root->vert_loc  = 0;
    root->vsum      = (scalar_t *) MALLOC(sizeof(index_t)*(root->n)*(root->tmax+1));

    //assert(tmax >= k-1);

    push_time();
    fprintf(stdout, "build query: ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j];
            index_t v = e[j+1];
            index_t t = e[j+2];
            index_t *pos_t = (pos + (n*t));
            if(start <= u && u <= stop) {
                // I am responsible for u, record adjacency to u
                pos_t[u]++;
            }
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                pos_t[v]++;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t *pos_t = pos + n*t;
        pos_t[u]++;
        pos_t[v]++;
    }
#endif

    index_t run = prefixsum(n*tmax, pos, 1);
    assert(run == (n*tmax+2*m));
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n*tmax; u++) {
            adj[pos[u]] = 0;
    }

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t th = 0; th < nt; th++) {
        index_t start = th*block_size;
        index_t stop = (th == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < 3*m; j+=3) {
            index_t u = e[j+0];
            index_t v = e[j+1];
            index_t t = e[j+2];
            if(start <= u && u <= stop) {
                // I am responsible for u, record adjacency to u
                index_t pu = pos[n*t+u];
                adj[pu + 1 + adj[pu]++] = v;
            }
            if(start <= v && v <= stop) {
                // I am responsible for v, record adjacency to v
                index_t pv = pos[n*t+v];
                adj[pv + 1 + adj[pv]++] = u;
            }
        }
    }
#else
    for(index_t j = 0; j < 3*m; j+=3) {
        index_t u = e[j+0];
        index_t v = e[j+1];
        index_t t = e[j+2];
        index_t pu = pos[n*t+u];
        index_t pv = pos[n*t+v];       
        adj[pu + 1 + adj[pu]++] = v;
        adj[pv + 1 + adj[pv]++] = u;
    }
#endif


    /*
    // TODO: works only for single source
    // update this part later
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < tmax; t++) {
        index_t *pos_t = (pos + (n*t));
        for(index_t i = 0; i < num_srcs; i++) {
            index_t s = sources[i];
            index_t ps = pos_t[s];
            adj[ps] = 0;
        }
    }
    */


    time = pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
    adjsort(n*tmax, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    // copy shades
    //push_time();
//#ifdef BUILD_PARALLEL
//#pragma omp parallel for
//#endif
//    for(index_t u = 0; u < n; u++) {
//        shade_map_t s = 0;
//        for(index_t j = 0; j < k; j++)
//            if(colors[u] == kk[j])
//                s |= 1UL << j;
//        shade[u] = s;
//        fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
//        //
//    }
//    time = pop_time();
//    fprintf(stdout, "[shade: %.2lf ms] ", time);
//    fflush(stdout);

    // copy resting time
    push_time();
    index_t *rest_time = g->rest_time;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        rtime[u] = rest_time[u];
    time = pop_time();
    fprintf(stdout, "[rtime: %.2lf ms] ", time);
    fflush(stdout);


    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    //print_temppathq(root);
    return root;
}

void update_sources_adj(index_t n, index_t tmax, index_t num_srcs, 
                        index_t *sources, index_t *pos, index_t *adj)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < tmax; t++) {
        index_t *pos_t = (pos + (n*t));
        for(index_t i = 0; i < num_srcs; i++) {
            index_t s = sources[i];
            index_t ps = pos_t[s];
            adj[ps] = 0;
        }
    }


}

void update_colors(index_t n, index_t k, index_t num_srcs, index_t *sources, 
                   index_t num_seps, index_t *separators, index_t *color)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        color[u] = 2;

    // currently handling single source
    for(index_t i = 0; i < num_srcs; i++) {
        index_t u = sources[i];
        color[u] = 1;
    }

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < num_seps; i++) {
        index_t u = separators[i];
        color[u] = 3;
    }
}

void get_motif_colors(index_t k, index_t *kk)
{
    kk[0] = 1;
    // not worth parallelising
    for(index_t i = 1; i < k; i++)
        kk[i] = 2;
}


void temppathq_update_shades(index_t k, index_t *kk, index_t *color, temppathq_t *root)
{
    shade_map_t *shade  = root->shade;
    index_t n           = root->n;
    root->k             = k;
    root->ns            = k;

    //update shades
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        shade_map_t s = 0;
        for(index_t j = 0; j < k; j++)
            if(color[u] == kk[j])
                s |= 1UL << j;
        shade[u] = s;
        //fprintf(stdout, "%4ld: 0x%08X\n", u, shade[u]);
    }
}


/****************************************************** Input reader (ASCII). */

void skipws(FILE *in)
{
    int c;
    do {
        c = fgetc(in);
        if(c == '#') {
            do {
                c = fgetc(in);
            } while(c != EOF && c != '\n');
        }
    } while(c != EOF && isspace(c));
    if(c != EOF)
        ungetc(c, in);
}

#define CMD_NOP              0
#define CMD_RUN_ORACLE       6
#define CMD_VLOC             7
#define CMD_VLOC_FINEGRAIN   8
#define CMD_EXHAUST_SEARCH   9

char *cmd_legend[] = { "no operation", "run oracle", "localised", "localised (fine-grained)", "exhaustive search"};

void reader_ascii(FILE *in, 
                  graph_t **g_out, 
                  index_t *num_srcs_out, index_t **sources_out,
                  index_t *num_seps_out, index_t **separators_out)
{
    push_time();
    push_memtrack();
    
    index_t n           = 0;
    index_t m           = 0;
    index_t tmax        = 0;
    index_t rtmax       = 0;
    index_t is_dir      = 0;
    graph_t *g          = (graph_t *) 0;
    index_t num_srcs    = 0;
    index_t *sources    = (index_t *) 0;
    index_t num_seps    = 0;
    index_t *separators = (index_t *) 0;
    index_t i, j, t, rt;
    skipws(in);
    while(!feof(in)) {
        skipws(in);
        int c = fgetc(in);
        switch(c) {
        case 'p':
            if(g != (graph_t *) 0)
                ERROR("duplicate parameter line");
            skipws(in);
            if(fscanf(in, "motif %ld %ld %ld %ld %ld", &n, &m, &tmax, &rtmax, &is_dir) != 5)
                ERROR("invalid parameter line");
            if(n <= 0 || m < 0 || tmax < 1 || rtmax < 1) {
                ERROR("invalid input parameters (n = %ld, m = %ld, tmax = %ld, rtmax = %ld)",
                       n, m, tmax, rtmax);
            }
            g = graph_alloc(n);
            graph_set_is_directed(g, is_dir);
            graph_set_max_time(g, tmax);
            graph_set_max_resttime(g, rtmax);
            break;
        case 'e':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before edges");
            skipws(in);
            if(fscanf(in, "%ld %ld %ld", &i, &j, &t) != 3)
                ERROR("invalid edge line");
            graph_add_edge(g, i-1, j-1, t-1);
            break;
        case 'r':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before motif");
            skipws(in);
            if(fscanf(in, "%ld %ld", &i, &rt) != 2)
                ERROR("invalid rest time line");
            if(i < 1 || i > n || rt < 1 || rt > rtmax)
                ERROR("invalid rest time line (u = %ld, rt = %ld with n = %ld and rtmax = %ld)", 
                      i, rt, n, rtmax);
            graph_set_resttime(g, i-1, rt);
            break;
        case 's':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before sources");
            skipws(in);
            if(fscanf(in, "%ld", &num_srcs) != 1)
                ERROR("invalid sources line");
            if(num_srcs < 1 || num_srcs > n)
                ERROR("invalid sources line (num-sources = %ld with n = %d)", num_srcs, n);
            if(num_srcs > 1)
                ERROR("current implementation only support single source (num-sources = %ld)", num_srcs);
            sources = alloc_idxtab(num_srcs);
            for(index_t i = 0; i < num_srcs; i++) {
                index_t s;
                skipws(in);
                if(fscanf(in, "%ld", &s) != 1)
                    ERROR("error parsing sources line");
                if(s < 1 || s > n)
                    ERROR("invalid sources line (s = %ld)", s);
                sources[i] = s-1;
            }
            break;
        case 't':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before separators");
            skipws(in);
            if(fscanf(in, "%ld", &num_seps) != 1)
                ERROR("invalid separators line");
            if(num_seps < 1 || num_seps > n)
                ERROR("invalid separators line (num-separators = %ld with n = %d)", num_seps, n);
            separators = alloc_idxtab(num_seps);
            for(index_t i = 0; i < num_seps; i++) {
                index_t s;
                skipws(in);
                if(fscanf(in, "%ld", &s) != 1)
                    ERROR("error parsing sources line");
                if(s < 1 || s > n)
                    ERROR("invalid separator (s = %ld)", s);
                separators[i] = s-1;
            }
            break;
        case EOF:
            break;
        default:
            ERROR("parse error");
        }
    }

    if(g == (graph_t *) 0)
        ERROR("no graph given in input");

    double time = pop_time();

    fprintf(stdout, 
            "input: n = %ld, m = %ld, t = %ld, rt = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            g->max_time,
            g->max_resttime,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");    
    fprintf(stdout, "sources [%ld]: ", num_srcs);
    for(index_t i = 0; i < num_srcs; i++)
        fprintf(stdout, " %ld", sources[i]+1);
    fprintf(stdout, "\n");
    fprintf(stdout, "separators [%ld]: ", num_seps);
    for(index_t i = 0; i < num_seps; i++)
        fprintf(stdout, "%ld", separators[i]+1);
    fprintf(stdout, "\n");
    
    *g_out          = g;
    *num_srcs_out   = num_srcs;
    *sources_out    = sources;
    *num_seps_out   = num_seps;
    *separators_out = separators;
}

/************************************************************ Temporal DFS. */

index_t temp_dfs(index_t n, index_t k, index_t tmax, index_t *pos,
                 index_t *adj, index_t *in_stack, index_t *rtime, 
                 index_t *reach_time, stk_t *s)
{
    // reached depth 'k'
    if(s->n == k) // TODO: fix this to s->n == k
        return 1;

    stack_node_t e;
    stack_top(s, &e);
    index_t u    = e.u;
    index_t l    = e.l;
    index_t min_t = e.t;
	index_t max_t = MIN(tmax, e.t + rtime[u]);

    for(index_t t = min_t; t <= max_t; t++) {
        index_t *pos_t = pos + (t-1)*n;
        index_t pu = pos_t[u];
        index_t nu = adj[pu];
        if(nu == 0) continue;
        index_t *adj_u = adj + pu;
        for(index_t i = 1; i <= nu; i++) {
            index_t v = adj_u[i];
            if(in_stack[v]) continue;

            stack_node_t e;
            e.u = v;
            e.l = l+1;
            e.t = t;
            stack_push(s, &e);
            in_stack[v] = 1;
			reach_time[v] = MIN(reach_time[v], t);

            temp_dfs(n, k, tmax, pos, adj, in_stack, rtime, reach_time, s);

            stack_pop(s, &e);
            in_stack[v] = 0;
        }
    }
    return 0; // not found
}


void exhaustive_search(temppathq_t *root, index_t src, index_t *reach_time)
{
    push_time();
    index_t n = root->n;
    index_t k = root->k;
    index_t tmax = root->tmax;
    index_t *pos = root->pos;
    index_t *adj = root->adj;
    index_t *rtime = root->rtime;
    index_t *in_stack = alloc_idxtab(n);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        in_stack[u] = 0;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        reach_time[u] = MATH_INF;

    rtime[src] = tmax;
    stk_t *s = stack_alloc(k);
    stack_node_t es;
    es.u = src;
    es.l = 0;
    es.t = 1;
    stack_push(s, &es);
    in_stack[src] = 1;
    double preproc_time = pop_time();

    push_time();
    temp_dfs(n, k, tmax, pos, adj, in_stack, rtime, reach_time, s);
    double dfs_time = pop_time();

    FREE(in_stack);
    stack_free(s);

    fprintf(stdout, " [%.2lfms %.2lfms %.2lfms]",  preproc_time, dfs_time, pop_time());
}


/**************************************************** get minimum timestamp. */

void vloc_min_timestamp(index_t n, index_t k, index_t *vloc_time, 
                             index_t *vloc_min_time)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t *vloc_time_u = vloc_time + (u*(k-1));
        index_t min_time = MATH_INF;
        for(index_t l = 2; l <= k; l++) {
            index_t cur_time = vloc_time_u[l-2];
            if(cur_time < min_time)
                min_time = cur_time;
        }
        vloc_min_time[u] = min_time;
    }
}

void vloc_table_out(FILE *out, index_t n, index_t k, index_t *vloc_time,
                    index_t *vloc_min_time, char *out_type)
{
    if(!strcmp(out_type, "csv")) {
        for(index_t u = 0; u < n; u++) {
            for(index_t l = 2; l <=k; l++) {
                index_t t_vloc = vloc_time[u*(k-1) + (l-2)];
                fprintf(out, "%ld;", t_vloc==MATH_INF?UNDEFINED:t_vloc);
            }
            fprintf(out, "%ld\n", vloc_min_time[u]==MATH_INF?UNDEFINED:vloc_min_time[u]);
            fflush(out);
        }
    } else {
        for(index_t u = 0; u < n; u++) {
            fprintf(out, "%5ld:", u+1);
            for(index_t l = 2; l <=k; l++) {
                //index_t t_vloc = vloc_time[(l-2)*n + u];
                index_t t_vloc = vloc_time[u*(k-1) + (l-2)];
                fprintf(out, " %6ld", t_vloc==MATH_INF?UNDEFINED:t_vloc);
            }
            fprintf(out, " %6ld\n", vloc_min_time[u]==MATH_INF?UNDEFINED:vloc_min_time[u]);
            fflush(out);
        }
    }
}

/****************************************************** program entry point. */
int main(int argc, char **argv)
{
    GF_PRECOMPUTE;

    push_time();
    push_memtrack();
    
    index_t arg_cmd         = CMD_NOP;
    index_t flag_help       = 0;
    index_t flag_test       = 0;
    index_t have_seed       = 0;
    index_t have_input      = 0;
    index_t have_output     = 0;
    index_t have_karg       = 0;
    index_t k_arg           = 0;
    index_t k               = 0;
    index_t seed            = 123456789;
    char *filename          = (char *) 0;
    char *filename_out      = (char *) 0;
    for(index_t f = 1; f < argc; f++) {
        if(argv[f][0] == '-') {
            if(!strcmp(argv[f], "-h") || !strcmp(argv[f], "-help")) { 
                flag_help = 1;
                break;
            }
            if(!strcmp(argv[f], "-oracle")) {
                arg_cmd = CMD_RUN_ORACLE;
            }
            if(!strcmp(argv[f], "-vloc")) {
                arg_cmd = CMD_VLOC;
            }
            if(!strcmp(argv[f], "-vloc-finegrain")) {
                arg_cmd = CMD_VLOC_FINEGRAIN;
            }
            if(!strcmp(argv[f], "-baseline")) {
                arg_cmd = CMD_EXHAUST_SEARCH;
            }
            if(!strcmp(argv[f], "-seed")) {
                if(f == argc - 1)
                    ERROR("random seed missing from command line");
                seed = atol(argv[++f]);
                have_seed = 1;
            }
            if(!strcmp(argv[f], "-k")) {
                if(f == argc -1)
                    ERROR("path length missing from command line");
                k_arg     = atol(argv[++f]);
                have_karg = 1;
            }
            if(!strcmp(argv[f], "-in")) {
                if(f == argc - 1)
                    ERROR("input file missing from command line");
                have_input  = 1;
                filename    = argv[++f];
            }
            if(!strcmp(argv[f], "-out")) {
                if(f == argc - 1)
                    ERROR("output file missing from command line");
                have_output     = 1;
                filename_out    = argv[++f];
            }
            if(!strcmp(argv[f], "-test")) {
                flag_test = 1;
            }
        }
    }

    fprintf(stdout, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stdout, " %s", argv[f]);
    fprintf(stdout, "\n");

    if(flag_help) {
        fprintf(stdout,
                "usage: %s -pre <value> -optimal -<command-type> -seed <value> -in <input-file> -<file-type> \n"
                "       %s -h/help\n"
                "\n"
                "  -<command-type>    : oracle 	           - decide existence of a solution\n"
                "                       vloc               - single run of vertex localisation\n"
                "                       vloc-finegrain     - fine-grained evaluation of the oracle\n"
                "                       baseline           - exhaustive-search algorithm\n"
                " -k <value>          : integer value in range 1 to n-1\n"
                " -seed <value>       : integer value in range 1 to 2^32 -1\n"
				"					    default value `%ld`\n"
                " -in <input-file>    : path to input file, `stdin` by default \n"
                " -out <output-file>  : path to output file, `reachability.out` by default \n"
                " -min                : reports minimum reachable time for each vertex to `output-file`\n"
                " -h or -help         : help\n"
                "\n"
                , argv[0], argv[0], seed);
        return 0;
    }

    if(have_seed == 0) {
        fprintf(stdout, 
                "no random seed given, defaulting to %ld\n", seed);
    }
    fprintf(stdout, "random seed = %ld\n", seed);
   
    FILE *in = stdin;
    if(have_input) {
        in = fopen(filename, "r");
        if(in == NULL) {
            ERROR("unable to open file '%s'", filename);
        } else {
            fprintf(stdout, "no input file specified, defaulting to stdin\n");
        }
        fflush(stdout);
    }

    FILE *out = stdout;
    if(have_output) {
        out = fopen(filename_out, "w");
        if(out == NULL)
            ERROR("unable to open file '%s'", filename_out);
    } else {
        out = fopen("reachability.out", "w");
        fprintf(stdout, "no output file specified, defaulting to `reachability.out`\n");
    }
    fflush(stdout);

    if(have_karg) {
        k = k_arg;
        fprintf(stdout, "path length specified in command line, changing to `k = %ld`\n", k);
    } else {
        k = 2;
        fprintf(stdout, "no path length specified, defaulting to `k = %ld`\n", k);
    }
    fflush(stdout);

    // initilize random number generator
    srand(seed); 

    index_t num_srcs;
    index_t num_seps;
    graph_t *g          = (graph_t *) 0;
    index_t *kk         = (index_t *) 0;
    index_t *sources    = (index_t *) 0;
    index_t *separators = (index_t *) 0;
    index_t *color      = (index_t *) 0;
    index_t cmd         = arg_cmd;  // by default execute command in input stream

    // read input graph : current implementation supports only ascii inputs
    reader_ascii(in, &g, &num_srcs, &sources, &num_seps, &separators);
    if(have_input) fclose(in); //close file-descriptor


    // build root query
    temppathq_t *root = (temppathq_t *) 0;
    if(g->is_directed) {
        root = build_temppathq_dir(g); 
    } else {
        root = build_temppathq(g);
    }
    graph_free(g); // free graph

    if(arg_cmd != CMD_EXHAUST_SEARCH) {
        // update adjacency list
        update_sources_adj(root->n, root->tmax, num_srcs, sources, root->pos, root->adj);

        // update vertex colors
        color = alloc_idxtab(root->n);
        update_colors(root->n, k, num_srcs, sources, num_seps, separators, color);
    }

    push_time();
    // execute command
    switch(cmd) {
    case CMD_NOP:
        {
            // no operation
            temppathq_free(root);
            break;
        }
    case CMD_RUN_ORACLE:
        {
            // --- run oracle ---
            fprintf(stdout, "oracle [temppath]: ");
            fflush(stdout);
            if(temppathq_execute(root))
                fprintf(stdout, " -- true\n");
            else
                fprintf(stdout, " -- false\n");
            temppathq_free(root);
        }
        break;
    case CMD_VLOC:
        {
             // --- run oracle ---
            fprintf(stdout, "oracle [temppath]: ");
            fflush(stdout);
            kk = alloc_idxtab(k);
            get_motif_colors(k, kk);
            temppathq_update_shades(k, kk, color, root);

            if(temppathq_execute(root))
                fprintf(stdout, " -- true\n");
            else
                fprintf(stdout, " -- false\n");

            scalar_t *vsum      = root->vsum;
            index_t n           = root->n;
            index_t tmax        = root->tmax;
            index_t *vloc_time  = (index_t *) MALLOC(sizeof(index_t)*n);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
            for(index_t u = 0; u < n; u++) {
                vloc_time[u] = UNDEFINED;
                index_t i_u0 = (u*(tmax+1));
                for(index_t i = 0; i <= tmax; i++) {
                    index_t i_ui = i_u0 + i;
                    if(vsum[i_ui]) {
                        vloc_time[u] = i;
                        break;
                    }
                }
            }

            fprintf(stdout, "time:\n");
            for(index_t u = 0; u < n; u++) {
                if(vloc_time[u] != UNDEFINED)
                    fprintf(stdout,"%10ld: %4ld\n", u+1, vloc_time[u]);
            }
            fflush(stdout);
            FREE(vloc_time);
            temppathq_free(root);
           
        }
        break;
    case CMD_VLOC_FINEGRAIN:
        {
            index_t n           = root->n;
            //index_t k           = root->k;
            index_t tmax        = root->tmax;
            index_t *vloc_time  = (index_t *) MALLOC(sizeof(index_t)*n*(k-1));

            kk = alloc_idxtab(k);
            for(index_t l = 2; l <= k; l++) {
                push_time();
                push_time(); // time update shades
                get_motif_colors(l, kk);
                root->k = l;
                temppathq_update_shades(l, kk, color, root);
                fprintf(stdout, "finegrained-oracle [%ld, shade:%.2fms]: ", l, pop_time());

                push_time(); // run oracle and time it
                //execute oracle
                if(temppathq_execute(root)) {
                    fprintf(stdout, " [%.2lf ms]-- true", pop_time());
                } else {
                    fprintf(stdout, " [%.2lf ms]-- false", pop_time());
                }

                push_time();
                scalar_t *vsum      = root->vsum;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
                for(index_t u = 0; u < n; u++) {
                    vloc_time[u*(k-1) + l-2] = MATH_INF;
                    index_t i_u0 = (u*(tmax+1));
                    for(index_t i = 0; i <= tmax; i++) {
                        index_t i_ui = i_u0 + i;
                        if(vsum[i_ui]) {
                            vloc_time[u*(k-1) + l-2] = i;
                            break;
                        }
                    }
                }
                fprintf(stdout, " [%.2lfms, %.2lfms]\n", pop_time(), pop_time());
                fflush(stdout);
            }

            index_t *vloc_min_time = alloc_idxtab(n);
            vloc_min_timestamp(n, k, vloc_time, vloc_min_time);

            if(flag_test) {
                for(index_t u = 0; u < n; u++) {
                    fprintf(out, "%ld\n", vloc_min_time[u]==MATH_INF?UNDEFINED:vloc_min_time[u]);
                }
                fflush(out);
            } else {
                if(have_output) {
                    vloc_table_out(out, n, k, vloc_time, vloc_min_time, "csv");
                } else {
                    vloc_table_out(out, n, k, vloc_time, vloc_min_time, "default");
                }
            }

            FREE(vloc_time);
            FREE(vloc_min_time);
            temppathq_free(root);
        }
        break;
    case CMD_EXHAUST_SEARCH:
        {
            push_time();
            index_t n = root->n;
            index_t *reach_time = alloc_idxtab(n);

            kk = alloc_idxtab(k);
            get_motif_colors(k, kk);
            root->k = k;

            fprintf(stdout, "exhaustive-search [%ld]:", k);

            exhaustive_search(root, sources[0], reach_time);

            push_time();
            if(have_output) {
                for(index_t u = 0; u < n; u++) {
                    fprintf(out, "%ld\n", reach_time[u]==MATH_INF?UNDEFINED:reach_time[u]);
                }
                fflush(out);
            } else {
                for(index_t u = 0; u < n; u++) {
                    fprintf(out, "%6ld\n", reach_time[u]==MATH_INF?UNDEFINED:reach_time[u]);
                }
                fflush(out);
            }

            // free memory
            FREE(reach_time);
            temppathq_free(root);

            fprintf(stdout, " [%.2lfms %.2lfms]\n", pop_time(), pop_time());
            fflush(stdout);
        }
        break;
    default:
        assert(0);
        break;
    }

    if(kk != (index_t *) 0) { FREE(kk); }
    if(sources != (index_t *) 0) { FREE(sources); }
    if(separators != (index_t *) 0) { FREE(separators); }
    if(color != (index_t *) 0) { FREE(color); }

    if(have_output) fclose(out); //clode file descriptor

    double cmd_time = pop_time();
    double time = pop_time();
    fprintf(stdout, "command done [ %.2lf ms %.2lf ms]\n", 
                    cmd_time, time);
    //if(input_cmd != CMD_NOP)
    //    FREE(cmd_args);

    //time = pop_time();
    fprintf(stdout, "grand total [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, "\n");
    fprintf(stdout, "host: %s\n", sysdep_hostname());
    fprintf(stdout, 
            "build: %s, %s, %s, %ld x %s\n",
#ifdef BUILD_PARALLEL
            "multithreaded",
#else
            "single thread",
#endif
#ifdef BUILD_PREFETCH
            "prefetch",
#else
            "no prefetch",
#endif
            GENF_TYPE,
            LIMBS_IN_LINE,
            LIMB_TYPE);
    fprintf(stdout, 
            "compiler: gcc %d.%d.%d\n",
            __GNUC__,
            __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
    fflush(stdout);
    assert(malloc_balance == 0);
    assert(memtrack_stack_top < 0);
    assert(start_stack_top < 0);

    return 0;
}
