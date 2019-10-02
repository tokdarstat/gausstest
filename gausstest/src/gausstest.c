#include "R.h"
#include "Rmath.h"
#include "R_ext/Applic.h"

const double pi = 3.14159265;
#define NR_END 1
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

// -------------------- UTILITY FUNCTIONS ---------------------------//

// a library of utility functions

double *vect(int n);
int *ivect(int n);
double **mymatrix(int nr, int nc);
double ****array4d(int n1, int n2, int n3, int n4);
double ***array3d(int n1, int n2, int n3);
void set_dvect(double *x, int n, double val);
void set_ivect(int *x, int n, int val);
void set_dmat(double **A, int nrow, int ncol, double val);
void set_darray3d(double ***A, int n1, int n2, int n3, double val);
void set_darray4d(double ****A, int n1, int n2, int n3, int n4, double val);
void copy_vec2vec(double *x, double *y, int n);
void copy_vec2mat(double *x, double **M, int nrow, int ncol, int byrow);
void copy_mat2vec(double **M, double *x, int nrow, int ncol, int byrow);
void copy_mat2mat(double **A, double **B, int nrow, int ncol);
void Rprintvec(char *a, char *format, double *x, int n);
void Rprintmat(char *a, char *format, double **x, int m, int n, int flip);
void Rprintveci(char *a, char *format, int *x, int n);
double inprod(double *x, double *y, int n);
double sum(double *x, int n);
double sumsquares(double *x, int n);
double pythag(double a, double b);
double lmvgammafn(double a, int p);
double vmax(double *x, int n);
double vmin(double *x, int n);
double logsum(double *lx, int n);
int rdraw(int n, double *prob, int inlog);
void rperm(int *v, int n, int k);

void set_lower_tri_zero(double **A, int n, int m);
void simple_cholesky(double **A, double **R, int N);
void cholesky_rank1_update(double **R, double *x, int N);
double logdet_UT(double **U, int n);
void trisolve(double **R, int m, double *b, double *x, int transpose);
void triprod(double **R, int m, int n, double *x, double *b, int transpose);
void tri_crossprod(double **U, double **A, int n);
double log_det_tri(double **U, int n);

void mat_sum(double **A, double **B, int nrow, int ncol, double **out);
void mat_times_mat(double **A, double **B, int ncomm, int nrow, int ncol, double **M); // A %*% B with dim(A) = c(nrow, ncomm), dim(B) = c(ncomm, ncol)
void mat_times_vec(double **A, double *x, int ncomm, int nrow, double *out); // A %*% x with dim(A) = c(nrow, ncomm), len(x) = ncomm
void tmat_times_mat(double **A, double **B, int ncomm, int nrow, int ncol, double **M); // t(A, B) with dim(A) = c(ncomm, nrow), dim(B) = c(ncomm, ncol)
void tmat_times_vec(double **A, double *x, int ncomm, int nrow, double *out); // t(A, x) with dim(A) = c(ncomm, nrow), len(x) = ncomm

double *vect(int n){
    return (double *)R_alloc(n, sizeof(double));
}
int *ivect(int n){
    return (int *)R_alloc(n, sizeof(int));
}
double **mymatrix(int nr, int nc){
    int i;
    double **m;
    m = (double **) R_alloc(nr, sizeof(double *));
    for (i = 0; i < nr; i++) m[i] = vect(nc);
    return m;
}
double ***array3d(int n1, int n2, int n3){
    int i;
    double ***a;
    a = (double ***)R_alloc(n1, sizeof(double **));
    for(i = 0; i < n1; i++) a[i] = mymatrix(n2, n3);
    return a;
}
double ****array4d(int n1, int n2, int n3, int n4){
    int i;
    double ****a;
    a = (double ****)R_alloc(n1, sizeof(double ***));
    for(i = 0; i < n1; i++) a[i] = array3d(n2, n3, n4);
    return a;
}
void set_dvect(double *x, int n, double val){
    int i;
    for(i = 0; i < n; i++) x[i] = val;
}
void set_dmat(double **A, int nrow, int ncol, double val){
    int i;
    for(i = 0; i < nrow; i++) set_dvect(A[i], ncol, val);
}
void set_darray3d(double ***A, int n1, int n2, int n3, double val){
    int i;
    for(i = 0; i < n1; i++) set_dmat(A[i], n2, n3, val);
}
void set_darray4d(double ****A, int n1, int n2, int n3, int n4, double val){
    int i;
    for(i = 0; i < n1; i++) set_darray3d(A[i], n2, n3, n4, val);
}
void set_ivect(int *x, int n, int val){
    int i;
    for(i = 0; i < n; i++) x[i] = val;
}
void copy_vec2vec(double *x, double *y, int n){
    int i;
    for(i = 0; i < n; i++) y[i] = x[i];
}
void copy_vec2mat(double *x, double **M, int nrow, int ncol, int byrow){
    int i, j, pos = 0;
    if(byrow){
        for(i = 0; i < nrow; i++){
            for(j = 0; j < ncol; j++){
                M[i][j] = x[pos];
                pos++;
            }
        }
    } else {
        for(j = 0; j < ncol; j++){
            for(i = 0; i < nrow; i++){
                M[i][j] = x[pos];
                pos++;
            }
        }
    }
}

void copy_mat2vec(double **M, double *x, int nrow, int ncol, int byrow){
    int i, j, pos = 0;
    if(byrow){
        for(i = 0; i < nrow; i++){
            for(j = 0; j < ncol; j++){
                x[pos] = M[i][j];
                pos++;
            }
        }
    } else {
        for(j = 0; j < ncol; j++){
            for(i = 0; i < nrow; i++){
                x[pos] = M[i][j];
                pos++;
            }
        }
    }
}

void copy_mat2mat(double **A, double **B, int nrow, int ncol){
    int i, j;
    for(i = 0; i < nrow; i++){
        for(j = 0; j < ncol; j++){
            B[i][j] = A[i][j];
        }
    }
}

void mat_sum(double **A, double **B, int nrow, int ncol, double **out){
    int i, j;
    for(i = 0; i < nrow; i++){
        for(j = 0; j < ncol; j++){
            out[i][j] = A[i][j] + B[i][j];
        }
    }
}
void mat_times_mat(double **A, double **B, int ncomm, int nrow, int ncol, double **M){
    int i, j, k;
    double val;
    for(i = 0; i < nrow; i++){
        for(j = 0; j < ncol; j++){
            val = 0.0;
            for(k = 0; k < ncomm; k++){
                val += A[i][k] * B[k][j];
            }
            M[i][j] = val;
        }
    }
}

void mat_times_vec(double **A, double *x, int ncomm, int nrow, double *out){
    int i, k;
    for(i = 0; i < nrow; i++){
        out[i] = 0.0;
        for(k = 0; k < ncomm; k++){
            out[i] += A[i][k] * x[k];
        }
        
    }
}


void tmat_times_mat(double **A, double **B, int ncomm, int nrow, int ncol, double **M){
    int i, j, k;
    double val;
    for(i = 0; i < nrow; i++){
        for(j = 0; j < ncol; j++){
            val = 0.0;
            for(k = 0; k < ncomm; k++){
                val += A[k][i] * B[k][j];
            }
            M[i][j] = val;
        }
    }
}

void tmat_times_vec(double **A, double *x, int ncomm, int nrow, double *out){
    int i, k;
    for(i = 0; i < nrow; i++){
        out[i] = 0.0;
        for(k = 0; k < ncomm; k++){
            out[i] += A[k][i] * x[k];
        }
        
    }
}


void Rprintvec(char *a, char *format, double *x, int n){
    int i;
    Rprintf("%s", a);
    for(i = 0; i < n; i++)
        Rprintf(format, x[i]);
    Rprintf("\n");
}
void Rprintmat(char *a, char *format, double **x, int m, int n, int flip){
    int i, j;
    Rprintf("%s\n", a);
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++)
            Rprintf(format, x[i][j]);
        Rprintf("\n");
    }
}
void Rprintveci(char *a, char *format, int *x, int n){
    int i;
    Rprintf("%s", a);
    for(i = 0; i < n; i++)
        Rprintf(format, x[i]);
    Rprintf("\n");
}
double inprod(double *x, double *y, int n){
    double ip = 0.0;
    int i;
    for(i = 0; i < n; i++)
        ip += x[i] * y[i];
    return ip;
}
double sum(double *x, int n){
    double a = 0.0;
    int i;
    for(i = 0; i < n; i++) a += x[i];
    return a;
}
double sumsquares(double *x, int n){
    return inprod(x, x, n);
}

/******************************************************************************/
double pythag(double a, double b)
/*******************************************************************************
 Computes (a2 + b2)1/2 without destructive underflow or overflow.
 *******************************************************************************/
{
    double absa,absb;
    absa=fabs(a);
    absb=fabs(b);
    if (absa > absb) return absa*sqrt(1.0+(absb/absa)*(absb/absa));
    else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa/absb)*(absa/absb)));
}

double lmvgammafn(double a, int p) {
    int pp = p * p - p, j;
    double lf = ((double)pp) * log(pi) / 4.0;
    for(j = 0; j < p; j++) lf += lgammafn(a - ((double)j) / 2.0);
    return lf;
}
double vmax(double *x, int n){
    int i;
    double xmax = x[0];
    for(i = 1; i < n; i++) if(x[i] > xmax) xmax = x[i];
    return xmax;
}
double vmin(double *x, int n){
    int i;
    double xmin = x[0];
    for(i = 1; i < n; i++) if(x[i] < xmin) xmin = x[i];
    return xmin;
}
double logsum(double *lx, int n){
    double lxmax = vmax(lx, n), a = 0.0;
    int i;
    for(i = 0; i < n; i++) a += exp(lx[i] - lxmax);
    return lxmax + log(a);
}
int rdraw(int n, double *prob, int inlog){
    double psum, u = runif(0.0, 1.0), cprob;
    int j = 0;
    
    if(inlog){
        psum = logsum(prob, n);
        cprob = exp(prob[0] - psum);
        while(u > cprob && j < n - 1){
            j++;
            cprob += exp(prob[j] - psum);
        }
    } else {
        psum = sum(prob, n);
        cprob = prob[0] / psum;
        while(u > cprob && j < n - 1){
            j++;
            if(prob[j] > 0.0) cprob += prob[j] / psum;
        }
    }
    return j;
}
void rperm(int *v, int n, int k) {
    
    int i, j, v2;
    for(i = 0; i < k; i++) {
        
        j = i + (int)floor((double)(n - i) * runif(0.0, 1.0));
        v2 = v[i];
        v[i] = v[j];
        v[j] = v2;
        
    }
    
}



// Triangular matrix manipulation, including Cholesky factorization //
// Default is to deal with upper triangular matrix //

void set_lower_tri_zero(double **A, int n, int m ){
    int i, j;
    for(i = 0; i < n; i++)
        for(j = i + 1; j < m; j++)
            A[j][i] = 0.0;
}

void simple_cholesky(double **A, double **R, int N){
    
    set_lower_tri_zero(R, N, N);
    int i, k, l;
    double u;
    for(i = 0; i < N; i++) R[i][i] = A[i][i];
    
    for(k = 0; k < N; k++){
        R[k][k] = sqrt(R[k][k]);
        for(i = k + 1; i < N; i++){
            u = A[i][k];
            for(R[k][i] = u, l = 0; l < k; l++){
                R[k][i] -= R[l][i] * R[l][k];
            }
            R[k][i] /= R[k][k];
            R[i][i] -= R[k][i] * R[k][i];
        }
    }
}

void cholesky_rank1_update(double **R, double *x, int N){
    // will rewrite both R and x
    int k, l;
    double r, c, s;
    for(k = 0; k < N; k++){
        if(x[k] != 0.0){
            r = sqrt(R[k][k]*R[k][k] + x[k]*x[k]);
            c = r / R[k][k];
            s = x[k] / R[k][k];
            R[k][k] = r;
            for(l = k + 1; l < N; l++){
                R[k][l] = (R[k][l] + s*x[l])/c;
                x[l] = c*x[l] - s*R[k][l];
            }
        }
    }
}


double logdet_UT(double **U, int n){
    double ld = 0.0;
    int i;
    for(i = 0; i < n; i++) ld += log(U[i][i]);
    return ld;
}

void trisolve(double **R, int m, double *b, double *x, int transpose){
    
    int i, j;
    if(transpose){
        for(j = 0; j < m; j++){
            for(x[j] = b[j], i = 0; i < j; i++)
                x[j] -= x[i] * R[i][j];
            x[j] /= R[j][j];
        }
    } else {
        for(j = m - 1; j >= 0; j--){
            for(x[j] = b[j], i = j + 1; i < m; i++)
                x[j] -= R[j][i] * x[i];
            x[j] /= R[j][j];
        }
    }
}

void triprod(double **R, int m, int n, double *x, double *b, int transpose){
    // when m = n, it is allowed to overwrite x with b
    // i.e., one may call the function as triprod(R, k, k, x, x)
    // to compute Rx and store it in x
    
    int i, j;
    double val;
    if(transpose){
        i = n;
        while(i > m){
            i--;
            for(val = 0.0, j = 0; j < m; j++) val += R[j][i] * x[j];
            b[i] = val;
        }
        while(i > 0){
            i--;
            for(val = 0.0, j = 0; j <= i; j++) val += R[j][i] * x[j];
            b[i] = val;
        }
    } else{
        for(i = 0; i < m; i++){
            for(val = 0.0, j = i; j < n; j++){
                val += R[i][j] * x[j];
            }
            b[i] = val;
        }
    }
}

void tri_crossprod(double **U, double **A, int n){
    int i, j, k;
    for(i = 0; i < n; i++){
        for(j = 0; j <= i; j++){
            A[i][j] = 0.0;
            for(k = 0; k <= j; k++) A[i][j] += U[k][i] * U[k][j];
            if(j < i) A[j][i] = A[i][j];
        }
    }
}

double log_det_tri(double **U, int n){
    double ld = 0.0;
    int i;
    for(i = 0; i < n; i++) ld += log(U[i][i]);
    return ld;
}

// eigenvalue decomposition //

/******************************************************************************/
void tred2(double **a, int n, double d[], double e[])
/*******************************************************************************
 Householder reduction of a real, symmetric matrix a[1..n][1..n].
 On output, a is replaced by the orthogonal matrix Q effecting the
 transformation. d[1..n] returns the diagonal elements of the tridiagonal matrix,
 and e[1..n] the off-diagonal elements, with e[1]=0. Several statements, as noted
 in comments, can be omitted if only eigenvalues are to be found, in which case a
 contains no useful information on output. Otherwise they are to be included.
 *******************************************************************************/
{
    int l,k,j,i;
    double scale,hh,h,g,f;
    
    for (i=n;i>=2;i--) {
        l=i-1;
        h=scale=0.0;
        if (l > 1) {
            for (k=1;k<=l;k++)
                scale += fabs(a[i-1][k-1]);
            if (scale == 0.0) /* Skip transformation. */
                e[i-1]=a[i-1][l-1];
            else {
                for (k=1;k<=l;k++) {
                    a[i-1][k-1] /= scale; /* Use scaled a's for transformation. */
                    h += a[i-1][k-1]*a[i-1][k-1]; /* Form sigma in h. */
                }
                f=a[i-1][l-1];
                g=(f >= 0.0 ? -sqrt(h) : sqrt(h));
                e[i-1]=scale*g;
                h -= f*g; /* Now h is equation (11.2.4). */
                a[i-1][l-1]=f-g; /* Store u in the ith row of a. */
                f=0.0;
                for (j=1;j<=l;j++) {
                    /* Next statement can be omitted if eigenvectors not wanted */
                    a[j-1][i-1]=a[i-1][j-1]/h; /* Store u/H in ith column of a. */
                    g=0.0; /* Form an element of A.u in g. */
                    for (k=1;k<=j;k++)
                        g += a[j-1][k-1]*a[i-1][k-1];
                    for (k=j+1;k<=l;k++)
                        g += a[k-1][j-1]*a[i-1][k-1];
                    e[j-1]=g/h; /* Form element of p in temporarily unused element of e. */
                    f += e[j-1]*a[i-1][j-1];
                }
                hh=f/(h+h); /* Form K, equation (11.2.11). */
                for (j=1;j<=l;j++) { /* Form q and store in e overwriting p. */
                    f=a[i-1][j-1];
                    e[j-1]=g=e[j-1]-hh*f;
                    for (k=1;k<=j;k++) /* Reduce a, equation (11.2.13). */
                        a[j-1][k-1] -= (f*e[k-1]+g*a[i-1][k-1]);
                }
            }
        } else
            e[i-1]=a[i-1][l-1];
        d[i-1]=h;
    }
    /* Next statement can be omitted if eigenvectors not wanted */
    d[1-1]=0.0;
    e[1-1]=0.0;
    /* Contents of this loop can be omitted if eigenvectors not
     wanted except for statement d[i]=a[i][i]; */
    for (i=1;i<=n;i++) { /* Begin accumulation of transformation matrices. */
        l=i-1;
        if (d[i-1]) { /* This block skipped when i=1. */
            for (j=1;j<=l;j++) {
                g=0.0;
                for (k=1;k<=l;k++) /* Use u and u/H stored in a to form P.Q. */
                    g += a[i-1][k-1]*a[k-1][j-1];
                for (k=1;k<=l;k++)
                    a[k-1][j-1] -= g*a[k-1][i-1];
            }
        }
        d[i-1]=a[i-1][i-1]; /* This statement remains. */
        a[i-1][i-1]=1.0; /* Reset row and column of a to identity matrix for next iteration. */
        for (j=1;j<=l;j++) a[j-1][i-1]=a[i-1][j-1]=0.0;
    }
}

/******************************************************************************/
void tqli(double d[], double e[], int n, double **z)
/*******************************************************************************
 QL algorithm with implicit shifts, to determine the eigenvalues and eigenvectors
 of a real, symmetric, tridiagonal matrix, or of a real, symmetric matrix
 previously reduced by tred2 sec. 11.2. On input, d[1..n] contains the diagonal
 elements of the tridiagonal matrix. On output, it returns the eigenvalues. The
 vector e[1..n] inputs the subdiagonal elements of the tridiagonal matrix, with
 e[1] arbitrary. On output e is destroyed. When finding only the eigenvalues,
 several lines may be omitted, as noted in the comments. If the eigenvectors of
 a tridiagonal matrix are desired, the matrix z[1..n][1..n] is input as the
 identity matrix. If the eigenvectors of a matrix that has been reduced by tred2
 are required, then z is input as the matrix output by tred2. In either case,
 the kth column of z returns the normalized eigenvector corresponding to d[k].
 *******************************************************************************/
{
    double pythag(double a, double b);
    int m,l,iter,i,k;
    double s,r,p,g,f,dd,c,b;
    
    for (i=2;i<=n;i++) e[i-1-1]=e[i-1]; /* Convenient to renumber the elements of e. */
    e[n-1]=0.0;
    for (l=1;l<=n;l++) {
        iter=0;
        do {
            for (m=l;m<=n-1;m++) { /* Look for a single small subdiagonal element to split the matrix. */
                dd=fabs(d[m-1])+fabs(d[m-1+1]);
                if ((double)(fabs(e[m-1])+dd) == dd) break;
            }
            if (m != l) {
                if (iter++ == 30) printf("Too many iterations in tqli");
                g=(d[l-1+1]-d[l-1])/(2.0*e[l-1]); /* Form shift. */
                r=pythag(g,1.0);
                g=d[m-1]-d[l-1]+e[l-1]/(g+SIGN(r,g)); /* This is dm - ks. */
                s=c=1.0;
                p=0.0;
                for (i=m-1;i>=l;i--) { /* A plane rotation as in the original QL, followed by Givens */
                    f=s*e[i-1];          /* rotations to restore tridiagonal form.                     */
                    b=c*e[i-1];
                    e[i-1+1]=(r=pythag(f,g));
                    if (r == 0.0) { /* Recover from underflow. */
                        d[i-1+1] -= p;
                        e[m-1]=0.0;
                        break;
                    }
                    s=f/r;
                    c=g/r;
                    g=d[i-1+1]-p;
                    r=(d[i-1]-g)*s+2.0*c*b;
                    d[i-1+1]=g+(p=s*r);
                    g=c*r-b;
                    /* Next loop can be omitted if eigenvectors not wanted */
                    for (k=1;k<=n;k++) { /* Form eigenvectors. */
                        f=z[k-1][i-1+1];
                        z[k-1][i-1+1]=s*z[k-1][i-1]+c*f;
                        z[k-1][i-1]=c*z[k-1][i-1]-s*f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l-1] -= p;
                e[l-1]=g;
                e[m-1]=0.0;
            }
        } while (m != l);
    }
}

double *eigen_vect_e;
void allocate_memory_eigen(int n){
    eigen_vect_e = vect(n);
}

void eigen(double **A, int n, double **Q, double *d){
    tred2(A, n, d, eigen_vect_e);
    copy_mat2mat(A, Q, n, n);
    tqli(d, eigen_vect_e, n, Q);
}


void QRfact(double **x, int nrow, int ncol, double **q, double **r){
    set_lower_tri_zero(r, ncol, ncol);
    int k, j, i;
    for(k = 0; k < ncol; k++){
        for(i = 0; i < nrow; i++) q[i][k] = x[i][k];
        for(j = 0; j < k; j++){
            r[j][k] = 0.0;
            for(i = 0; i < nrow; i++) r[j][k] += q[i][j] * q[i][k];
            for(i = 0; i < nrow; i++) q[i][k] -= r[j][k] * q[i][j];
        }
        r[k][k] = 0.0;
        for(i = 0; i < nrow; i++) r[k][k] += q[i][k] * q[i][k];
        r[k][k] = sqrt(r[k][k]);
        for(i = 0; i < nrow; i++) q[i][k] /= r[k][k];
    }
}


// ---------------------------- END UTILITY ----------------------------//

double *scratch_dvect_p, **scratch_dmatrix_11;
int *scratch_ivect_p;

void cholesky_identity_update(double **R, int p, double **R2){
    int i;
    copy_mat2mat(R, R2, p, p);
    for(i = 0; i < p; i++){
        set_dvect(scratch_dvect_p, p, 0.0);
        scratch_dvect_p[i] = 1.0;
        cholesky_rank1_update(R2, scratch_dvect_p, p);
    }
}


void allocate_memory_general(int p){
    scratch_dvect_p = vect(p);
    scratch_dmatrix_11 = mymatrix(1,1);
    scratch_ivect_p = ivect(p);
}

// SIMULATE FROM WISHART DISTRIBUTION
// RETURNS ONLY THE UPPER TRIANGULAR CHOLESKY FACTOR
// APPROACH: U = STANDARD BARTLETT DECOMP;
//           return U %*% R = t(t(R) %*% t(U))
//           where R = UTchol_in
// REQUIRES MEMORY ALLOCATION TO: scratch_dvect_p

double rwishart_UT(int p, double df, int scaled, double **UTchol_in, double **UTchol_out, int get_logdens){
    
    int i, j;
    double ld = 0.0;
    
    for(i = 0; i < p; i++){
        UTchol_out[i][i] = sqrt(rchisq(df - (double)i));
        for(j = 0; j < i; j++){
            UTchol_out[i][j] = 0.0;
            UTchol_out[j][i] = rnorm(0.0, 1.0);
        }
    }
    double tr = 0.0;
    for(i = 0; i < p; i++) tr += sumsquares(UTchol_out[i] + i, p - i);
    
    if(scaled){
        for(i = 0; i < p; i++){
            triprod(UTchol_in, p, p, UTchol_out[i], UTchol_out[i], /*transpose =*/ 1);
        }
    }
    if(get_logdens){
        ld -= 0.5 * (double)p * df * log(2.0) + lmvgammafn(df / 2.0, p) + 0.5 * tr;
        ld += (df - (double)(p+1)) * logdet_UT(UTchol_out, p);
        if(scaled) ld -= df * logdet_UT(UTchol_in, p);
    }
    return ld;
}


double dwishart_UT(double **X_UT, int p, double df, int scaled, double **scale_mat_UT, int islog){
    
    int i;
    double ld = 0.0, tr = 0.0;
    
    if(scaled){
        for(i = 0; i < p; i++){
            trisolve(scale_mat_UT, p, X_UT[i], scratch_dvect_p, /*transpose =*/ 1);
            tr += sumsquares(scratch_dvect_p, p);
        }
        ld -= df * logdet_UT(scale_mat_UT, p) + 0.5 * tr;
    } else {
        for(i = 0; i < p; i++) tr += sumsquares(X_UT[i] + i, p - i);
        ld -= 0.5 * tr;
    }
    ld -= 0.5 * (double)p * df * log(2.0) + lmvgammafn(df / 2.0, p);
    ld += (df - (double)(p+1)) * logdet_UT(X_UT, p);
    if(!islog) ld = exp(ld);
    return ld;
}


// SIMULATE FROM MULTIVARIATE BETA DISTRIBUTION WITH PARAMETERS omega1 and omega2
// RETURNS ONLY THE UPPER TRIANGULA CHOLESKY FACTOR
// APPROACH: A ~ Wishart(2*omega1); B ~ Wishart(2*pmega2);
//           A + B = U'U; A = R'R; return R %*% inv(U) = t(backsolve(U, t(R), transpose = TRUE))
// REQUIRES MEMORY ALLOCATION TO: rmvbeta_A_UTchol, rmvbeta_B_UTchol, rmvbeta_A, rmvbeta_B,
//                                rmvbeta_A_plus_B, rmvbeta_U, scratch_matrix_11

double **rmvbeta_A_UTchol, **rmvbeta_B_UTchol, **rmvbeta_A, **rmvbeta_B, **rmvbeta_A_plus_B, **rmvbeta_U;

void allocate_memory_rmvbeta(int p){
    rmvbeta_A_UTchol = mymatrix(p, p);
    rmvbeta_B_UTchol = mymatrix(p, p);
    rmvbeta_A = mymatrix(p, p);
    rmvbeta_B = mymatrix(p, p);
    rmvbeta_A_plus_B = mymatrix(p, p);
    rmvbeta_U = mymatrix(p, p);
}

void rmvbeta_UT(int p, double omega1, double omega2, double **rmvbeta_out_UTchol){
    
    rwishart_UT(p, 2.0 * omega1, 0, scratch_dmatrix_11, rmvbeta_A_UTchol, 0);
    rwishart_UT(p, 2.0 * omega2, 0, scratch_dmatrix_11, rmvbeta_B_UTchol, 0);
    int i;
    tri_crossprod(rmvbeta_A_UTchol, rmvbeta_A, p);
    tri_crossprod(rmvbeta_B_UTchol, rmvbeta_B, p);
    mat_sum(rmvbeta_A, rmvbeta_B, p, p, rmvbeta_A_plus_B);
    simple_cholesky(rmvbeta_A_plus_B, rmvbeta_U, p);
    set_dmat(rmvbeta_out_UTchol, p, p, 0.0);
    for(i = 0; i < p; i++) trisolve(rmvbeta_U, p, rmvbeta_A_UTchol[i], rmvbeta_out_UTchol[i], 1);
}

double *mvnorm_z;
void allocate_memory_mvnorm(int n){
    mvnorm_z = vect(n);
}

double rmvnorm(int len, double *mean_vec, double **sqrt_inv_cov, double var_factor, double *out, int get_logdens){
    int i;
    double ld = 0.0, sd_factor = sqrt(var_factor);
    
    for(i = 0; i < len; i++) mvnorm_z[i] = rnorm(0.0, sd_factor);
    trisolve(sqrt_inv_cov, len, mvnorm_z, out, 0);
    for(i = 0; i < len; i++) out[i] += mean_vec[i];
    if(get_logdens){
        ld = -0.5 * (double)len * log(2.0*pi*var_factor) + logdet_UT(sqrt_inv_cov, len) - 0.5 * sumsquares(mvnorm_z, len) / var_factor;
    }
    return ld;
}

double dmvnorm(double *x, int len, double *mean_vec, double **sqrt_inv_cov, double var_factor, int islog){
    int i;
    double sd_factor = sqrt(var_factor);
    for(i = 0; i < len; i++) mvnorm_z[i] = (x[i] - mean_vec[i]) / sd_factor;
    triprod(sqrt_inv_cov, len, len, mvnorm_z, mvnorm_z, 0);
    
    double ld = -0.5 * (double)len * log(2.0*pi*var_factor) + logdet_UT(sqrt_inv_cov, len) - 0.5 * sumsquares(mvnorm_z, len);
    if(!islog) ld = exp(ld);
    return ld;
}

// ------- MAIN FUNCTIONS ---------- //

void test_function(int *dim, double *pars, double *mu, double *S, double *V, double *x){
    int p = dim[0];
    double df = pars[0];
    double omega1 = pars[1];
    double omega2 = pars[2];
    double var_factor = pars[3];
    //Rprintf("p = %d, df = %g, omega1 = %g, omega2 = %g\n", p, df, omega1, omega2);
    
    double **Smat = mymatrix(p, p);
    copy_vec2mat(S, Smat, p, p, /*byrow = */ 0);
    //Rprintmat("Smat = ", "%g ", Smat, p, p, 1);
    
    double **R = mymatrix(p, p);
    simple_cholesky(Smat, R, p);
    //Rprintmat("R = ", "%g ", R, p, p, 1);
    
    allocate_memory_general(p);
    
    double **W = mymatrix(p, p);
    GetRNGstate();
    double ld_wish = rwishart_UT(p, df, 1, R, W, 1);
    Rprintf("ld_wish = %g, dwishart = %g\n\n", ld_wish, dwishart_UT(W, p, df, 1, R, 1));
    
    tri_crossprod(W, Smat, p);
    //Rprintmat("Wishart draw = ", "%g ", Smat, p, p, 1);
    copy_mat2vec(Smat, S, p, p, 0);
    
    allocate_memory_rmvbeta(p);
    rmvbeta_UT(p, omega1, omega2, Smat);
    
    allocate_memory_mvnorm(p);
    double ld_norm = rmvnorm(p, mu, Smat, var_factor, x, 1);
    Rprintf("ld_norm = %g, dmvnorm = %g\n\n", ld_norm, dmvnorm(x, p, mu, Smat, var_factor, 1));
    
    double **Vmat = mymatrix(p, p);
    tri_crossprod(Smat, Vmat, p);
    copy_mat2vec(Vmat, V, p, p, 0);
    
    
    double **VQ = mymatrix(p, p);
    double *Vd = vect(p);
    allocate_memory_eigen(p);
    Rprintmat("V = ", "%g ", Vmat, p, p, 1);
    eigen(Vmat, p, VQ, Vd);
    Rprintvec("V eigen values = ", "%g ", Vd, p);
    Rprintmat("V eigen vectors = ", "%g ", VQ, p, p, 0);
    
    double **QQ = mymatrix(p, p);
    tmat_times_mat(VQ, VQ, p, p, p, QQ);
    Rprintmat("Q'Q = ", "%2.4f ", QQ, p, p, 0);
    
    PutRNGstate();
}

// THE SEQUENTIAL P-DIMENSIONAL IMPORTANCE SAMPLER //



void cluster_update(int p, int nsub, double **w, double k, double **d, double **sum_w, double **sum_wsq, double **d1, double **d2, double *log_q){
    int i, sub;
    double dee, dtilde, mean_term, mean_term_sq, var_term, log_q_logsum;
    for(sub = 0; sub < nsub; sub++){
        for(i = 0; i < p; i++){
            dee = d[sub][i];
            dtilde = dee + k * (1.0 - dee);
            d1[sub][i] = dee * (1.0 + k * (1.0 - dee)) / dtilde;
            d2[sub][i] = (1.0 - dee) / dtilde;
        }
        
        log_q[sub] = 0.0;
        if(k == 1.0) {
            copy_vec2vec(w[sub], sum_w[sub], p);
            for(i = 0; i < p; i++) sum_wsq[sub][i] = w[sub][i]*w[sub][i];
        } else {
            for(i = 0; i < p; i++){
                sum_w[sub][i] += w[sub][i];
                sum_wsq[sub][i] += w[sub][i]*w[sub][i];
                mean_term = sum_w[sub][i] / k;
                mean_term_sq = mean_term * mean_term;
                var_term = sum_wsq[sub][i] / k - mean_term_sq;
                
                dee = d[sub][i];
                dtilde = dee + k * (1.0 - dee);

                log_q[sub] -= 0.5 * (k * (var_term/dee + mean_term_sq/dtilde) + (k-1.0)*log(dee) + log(dtilde));
            }
        }
    }
    //log_q[0] = 0.0;
    //for(sub = 1; sub < nsub; sub++) log_q[sub] = -100.0;
    log_q_logsum = logsum(log_q, nsub);
    for(sub = 0; sub < nsub; sub++) log_q[sub] -= log_q_logsum;

    //Rprintvec("log_q = ", "%g ", log_q, nsub);
}

double ****VQ, ***Vd, ***Vd1, ***Vd2;
double **yMat, **zMat, **Vmat, **sqrt_Vmat, ***w_curr, *z_curr;
double *lpclust, *lp_sub;
double *clust_size, ***clust_sum_w, ***clust_sum_wtw, **clust_log_q;

void allocate_memory_seqsamp_mv(int n, int p, int nsub){
    VQ = array4d(n, nsub, p, p);
    Vd = array3d(n, nsub, p);
    Vd1 = array3d(n, nsub, p);
    Vd2 = array3d(n, nsub, p);
    yMat = mymatrix(n, p);
    zMat = mymatrix(n, p);
    Vmat = mymatrix(p, p);
    sqrt_Vmat = mymatrix(p, p);
    w_curr = array3d(n, nsub, p);
    z_curr = vect(p);
    lpclust = vect(n);
    lp_sub = vect(nsub);
    clust_size = vect(n);
    clust_sum_w = array3d(n, nsub, p);
    clust_sum_wtw = array3d(n, nsub, p);
    clust_log_q = mymatrix(n, nsub);
}

int mvbeta_base;
void get_atoms(int p, double omega1, double omega2, double **Q, double *D){
    if(p > 1){
        if(mvbeta_base){
            rmvbeta_UT(p, omega1, omega2, sqrt_Vmat); // rmvbeta draw as a UT cholesky factor
            tri_crossprod(sqrt_Vmat, Vmat, p);        // the actual draw
            eigen(Vmat, p, Q, D);                     // and its eigen value and vectors
        } else {
            int i, j;
            for(i = 0; i < p; i++) for(j = 0; j < p; j++) Vmat[i][j] = rnorm(0.0,1.0);
            QRfact(Vmat, p, p, Q, sqrt_Vmat);
            for(i = 0; i < p; i++) D[i] = rbeta(omega1, omega2);
        }
    } else {
        Q[0][0] = 1.0;
        D[0] = rbeta(omega1, omega2);
    }
}

double seqsamp_mv(double *mu, double **inv_sqrt_sig, double alpha, double omega1, double omega2, int n, int p, int nsub, int *perm, int *nclust) {

    int i, j, s, t, sub;
    int clust_num, clust_pick;
    double lpnorm = 0.0, lalpha = log(alpha);
    double wgap;
    
    // set clust_sum_w[][] and clust_size[] to zero
    set_dvect(clust_size, n, 0.0);
    set_darray3d(clust_sum_w, n, nsub, p, 0.0);
    set_darray3d(clust_sum_wtw, n, nsub, p, 0.0);
    
    // Get zMat with zMat[i] = triprod(inv_sqrt_sig, yMat[i] - mu, transpose = TRUE)
    for(i = 0; i < n; i++){
        s = perm[i];
        copy_vec2vec(yMat[i], scratch_dvect_p, p);
        for(j = 0; j < p; j++) scratch_dvect_p[j] -= mu[j];
        triprod(inv_sqrt_sig, p, p, scratch_dvect_p, zMat[s], 1);
    }
    // First cluster with first observation (modulo permutation)
    
    clust_num = 1;
    clust_pick = 0;
    clust_size[clust_pick] = 1.0;
    copy_vec2vec(zMat[clust_pick], z_curr, p);
    double lwt = -0.5*sumsquares(z_curr, p);

    for(sub = 0; sub < nsub; sub++){
        get_atoms(p, omega1, omega2, VQ[clust_pick][sub], Vd[clust_pick][sub]);
        tmat_times_vec(VQ[clust_pick][sub], z_curr, p, p, w_curr[clust_pick][sub]);
    }
    
    cluster_update(p, nsub, w_curr[clust_pick], clust_size[clust_pick], Vd[clust_pick], clust_sum_w[clust_pick], clust_sum_wtw[clust_pick], Vd1[clust_pick], Vd2[clust_pick], clust_log_q[clust_pick]);
    
    for(s = 1; s < n; s++) {
        
        copy_vec2vec(zMat[s], z_curr, p);
        
        for(t = 0; t < clust_num; t++){
            for(sub = 0; sub < nsub; sub++){
                lp_sub[sub] = clust_log_q[t][sub];
                tmat_times_vec(VQ[t][sub], z_curr, p, p, w_curr[t][sub]);

                for(j = 0; j < p; j++){
                    wgap = w_curr[t][sub][j] - Vd2[t][sub][j] * clust_sum_w[t][sub][j];
                    lp_sub[sub] += 0.5*(w_curr[t][sub][j]*w_curr[t][sub][j] - wgap*wgap / Vd1[t][sub][j]) - 0.5*log(Vd1[t][sub][j]);
                }
            }
            lpclust[t] = log(clust_size[t]) + logsum(lp_sub, nsub) - 0.5*sumsquares(z_curr, p);
        }
        lpclust[clust_num] = lalpha - 0.5*sumsquares(z_curr, p);
        lpnorm = logsum(lpclust, clust_num + 1);
        lwt += lpnorm - log(alpha + (double)s);

        clust_pick = rdraw(clust_num + 1, lpclust, 1);
        
        if(clust_pick == clust_num) { // Create a new cluster for the current observation
            for(sub = 0; sub < nsub; sub++){
                get_atoms(p, omega1, omega2, VQ[clust_pick][sub], Vd[clust_pick][sub]);
                tmat_times_vec(VQ[clust_pick][sub], z_curr, p, p, w_curr[clust_pick][sub]);
            }
            
            clust_num++;
            clust_size[clust_pick] = 1.0;
        } else { // Add current observation to an existing cluster, and update
            clust_size[clust_pick] += 1.0;
        }
        cluster_update(p, nsub, w_curr[clust_pick], clust_size[clust_pick], Vd[clust_pick], clust_sum_w[clust_pick], clust_sum_wtw[clust_pick], Vd1[clust_pick], Vd2[clust_pick], clust_log_q[clust_pick]);
        
    }
    nclust[0] = clust_num;
    double dn = (double)n, dp = (double)p;
    return lwt + dn * log_det_tri(inv_sqrt_sig, p) - 0.5*dn*dp*log(2.0*pi);
}

// Sample from the posterior of (mu, U), under ther right Haar prior
// for the MV Gaussian model
// Y[i] ~ N(mean = mu, inv_cov = UU') where U is upper-triangular,
// Note that U is NOT the Cholesky factor of inv_cov!

double *ybar, **inv_sqrt_var, **inv_sqrt_scale, **inv_sqrt_sse;
double **imp_samp_mvF_mvT_lambda, *imp_samp_mvF_mvT_T;
double **imp_samp_matF_mvT_lambda, **imp_samp_matF_mvT_omega, **imp_samp_matF_mvT_delta, **imp_samp_matF_mvT_delta_plus, *imp_samp_matF_mvT_T;


void allocate_memory_imp_samp_mvF_mvT(int p){
    ybar = vect(p);
    inv_sqrt_var = mymatrix(p, p);
    inv_sqrt_scale = mymatrix(p, p);
    imp_samp_mvF_mvT_lambda = mymatrix(p, p);
    imp_samp_mvF_mvT_T = vect(p);
}

void allocate_memory_imp_samp_matF_mvT(int p){
    ybar = vect(p);
    inv_sqrt_var = mymatrix(p, p);
    inv_sqrt_scale = mymatrix(p, p);
    imp_samp_matF_mvT_lambda = mymatrix(p, p);
    imp_samp_matF_mvT_omega = mymatrix(p, p);
    imp_samp_matF_mvT_delta = mymatrix(p, p);
    imp_samp_matF_mvT_delta_plus = mymatrix(p, p);
    imp_samp_matF_mvT_T = vect(p);
}




double imp_samp_mvF_mvT(double *mu, double **inv_sqrt_sig, int p, int n, double df_imp, double scale_inflation, double *ldet_term, double *trace_term, double *tt_term){
    int i, j;
    double dp = (double)p, inv_sqrt_n = 1.0 / sqrt((double)n);
    double log_density = 0.0;

    // Sigma scaling to mvF
    double psi_sqrt = sqrt(rchisq(df_imp));

    // get inv_sqrt_sig and add log density part to log_density
    set_dmat(imp_samp_mvF_mvT_lambda, p, p, 0.0);
    for(j = 0; j < p; j++){
        imp_samp_mvF_mvT_lambda[j][j] = sqrt(rchisq(df_imp - dp + (double)(j+1))) / psi_sqrt;
        for(i = 0; i < j; i++) imp_samp_mvF_mvT_lambda[i][j] = rnorm(0.0, 1.0) / psi_sqrt;
    }
    for(i = 0; i < p; i++){
        triprod(imp_samp_mvF_mvT_lambda, p, p, inv_sqrt_scale[i], inv_sqrt_sig[i], 1);
    }
    

    double log_det_lam = log_det_tri(imp_samp_mvF_mvT_lambda, p);
    double trace_lam_lamt = 0.0;
    for(i = 0; i < p; i++) trace_lam_lamt += sumsquares(imp_samp_mvF_mvT_lambda[i], p);

    log_density += (df_imp + dp + 1.0)*log_det_lam - 0.5*df_imp*(dp+1.0)*log1p(trace_lam_lamt);

    // get mu and add log density part to log_density
    double z_scale = scale_inflation * inv_sqrt_n;
    double phi_sqrt = sqrt(rchisq(df_imp)/df_imp);
    for(j = 0; j < p; j++) imp_samp_mvF_mvT_T[j] = rnorm(0.0, 1.0) / phi_sqrt;
    double sumsq_T = sumsquares(imp_samp_mvF_mvT_T, p);
    log_density += log_det_lam - 0.5*(df_imp+dp)*log1p(sumsq_T / df_imp);
    
    trisolve(inv_sqrt_sig, p, imp_samp_mvF_mvT_T, scratch_dvect_p, 1);
    for(j = 0; j < p; j++) mu[j] = ybar[j] + z_scale * scratch_dvect_p[j];

    ldet_term[0] = log_det_lam;
    trace_term[0] = trace_lam_lamt;
    tt_term[0] = sumsq_T;
    return log_density;
}

double imp_samp_matF_mvT(double *mu, double **inv_sqrt_sig, int p, int n, double df_imp, double scale_inflation, double *ldet_term, double *trace_term, double *tt_term){
    int i, j;
    double dp = (double)p, inv_sqrt_n = 1.0 / sqrt((double)n);
    double log_density = 0.0;
    
    // Sigma scaling Psi = crossprod(Omega)
    set_dmat(imp_samp_matF_mvT_omega, p, p, 0.0);
    for(j = 0; j < p; j++){
        imp_samp_matF_mvT_omega[j][j] = sqrt(rchisq(df_imp - (double)j));
        for(i = 0; i < j; i++) imp_samp_matF_mvT_omega[i][j] = rnorm(0.0, 1.0);
    }
    
    // get inv_sqrt_sig and add log density part to log_density
    set_dmat(imp_samp_matF_mvT_lambda, p, p, 0.0);
    for(j = 0; j < p; j++){
        imp_samp_matF_mvT_lambda[j][j] = sqrt(rchisq(df_imp - dp + (double)(j+1)));
        for(i = 0; i < j; i++) imp_samp_matF_mvT_lambda[i][j] = rnorm(0.0, 1.0);
    }
    
    set_dmat(imp_samp_matF_mvT_delta, p, p, 0.0);
    for(j = 0; j < p; j++){
        set_dvect(scratch_dvect_p, p, 0.0);
        scratch_dvect_p[j] = 1.0;
        trisolve(imp_samp_matF_mvT_omega, p, scratch_dvect_p, scratch_dvect_p, 1); //j-th column of solve(t(omega))
        triprod(imp_samp_matF_mvT_lambda, p, p, scratch_dvect_p, imp_samp_matF_mvT_delta[j], 1); //j-th row of delta
    }
    
    set_dmat(imp_samp_matF_mvT_delta_plus, p, p, 0.0);
    cholesky_identity_update(imp_samp_matF_mvT_delta, p, imp_samp_matF_mvT_delta_plus);
    
    for(i = 0; i < p; i++){
        triprod(imp_samp_matF_mvT_delta, p, p, inv_sqrt_scale[i], inv_sqrt_sig[i], 1);
    }
    
    
    double log_det_del = log_det_tri(imp_samp_matF_mvT_delta, p);
    double log_det_deltdel_plus_id = 2.0*log_det_tri(imp_samp_matF_mvT_delta_plus, p);
    double trace_del_delt = 0.0;
    for(i = 0; i < p; i++) trace_del_delt += sumsquares(imp_samp_matF_mvT_delta[i], p);
    
    log_density += (df_imp + dp + 1.0)*log_det_del - df_imp*log_det_deltdel_plus_id;
    
    // get mu and add log density part to log_density
    double z_scale = scale_inflation * inv_sqrt_n;
    double phi_sqrt = sqrt(rchisq(df_imp)/df_imp);
    for(j = 0; j < p; j++) imp_samp_matF_mvT_T[j] = rnorm(0.0, 1.0) / phi_sqrt;
    double sumsq_T = sumsquares(imp_samp_matF_mvT_T, p);
    log_density += log_det_del - 0.5*(df_imp+dp)*log1p(sumsq_T / df_imp);
    
    trisolve(inv_sqrt_sig, p, imp_samp_matF_mvT_T, scratch_dvect_p, 1);
    for(j = 0; j < p; j++) mu[j] = ybar[j] + z_scale * scratch_dvect_p[j];
    
    ldet_term[0] = log_det_del;
    trace_term[0] = trace_del_delt;
    tt_term[0] = sumsq_T;
    return log_density;
}

double null_log_likelihood(double *mu, double **inv_sqrt_sig, int n, int p){
    int i, j;
    double dn=(double)n, dp=(double)p;
    for(i = 0; i < n; i++){
        copy_vec2vec(yMat[i], scratch_dvect_p, p);
        for(j = 0; j < p; j++) scratch_dvect_p[j] -= mu[j];
        triprod(inv_sqrt_sig, p, p, scratch_dvect_p, zMat[i], 1);
    }
    double log_likelihood = dn * log_det_tri(inv_sqrt_sig, p) - 0.5*dn*dp*log(2.0*pi);
    for(i = 0; i < n; i++) log_likelihood -= 0.5 * sumsquares(zMat[i], p);
    
    return log_likelihood;
}

// MAIN FUNCTION THAT INTERFACES WITH THE R WRAPPER TO COMPUTE BAYES FACTOR//
// INPUT:
// y: data matrix (nxp) in the form of a long vector of length n*p
// mean_y: vector of columne means of the data matrix
// sse_y: (n - 1) times the covariance matrix presented as a long cector
// inv_sse_y: inverse of sse_y presented as a long vector
// dims: integer vector giving problem dimensions
// hpar: hyperparameters of the model
// lwt: vector to retrieve the output (log importance weights)

//void dpm_gof_mv(double *y, double *mean_y, double *sse_y, int *dims, double *hpar, double *lwt, double *sso) {
void dpm_gof_mv(double *y, double *mean_y, double *inv_sqrt_var_y, double *inv_sqrt_scale_var, int *dims, double *hpar, double *lmarg_null, double *lwt_null, double *lwt_alt, /*double *lwt_ratio,*/ double *sso, double *limpd, double *mu_store, double *inv_sqrt_sigma_store, int *nclust_store, int *use_mvbeta_as_base) {

    int s;
    int n = dims[0], p = dims[1], nmc = dims[2], to_store = dims[3], nsub = dims[4];
    double alpha = hpar[0], omega1 = hpar[1], omega2 = hpar[2], df_imp = hpar[3], scale_inflation = hpar[4];
    //double scale_inflation_sq = scale_inflation * scale_inflation;
    mvbeta_base = use_mvbeta_as_base[0];
    
    double seqsamp_out;
    double *mu_samp = vect(p);
    double **inv_sqrt_sig_samp = mymatrix(p, p);
    int *perm = ivect(n);
    for(s = 0; s < n; s++) perm[s] = s;
    
    // globally defined pointers for clustering
    allocate_memory_general(p);
    allocate_memory_eigen(p);
    allocate_memory_seqsamp_mv(n, p, nsub);
    allocate_memory_rmvbeta(p);
    allocate_memory_mvnorm(p);
    //allocate_memory_imp_samp_mvF_mvT(p);
    allocate_memory_imp_samp_matF_mvT(p);

    copy_vec2mat(y, yMat, n, p, 0);
    copy_vec2vec(mean_y, ybar, p); // p-vector of means
    copy_vec2mat(inv_sqrt_var_y, inv_sqrt_var, p, p, 0);
    copy_vec2mat(inv_sqrt_scale_var, inv_sqrt_scale, p, p, 0);
    
    double dn = (double)n, dp = (double)p;
    double log_det_var_y = -2.0*log_det_tri(inv_sqrt_var, p);
    double log_det_scale_var = -2.0*log_det_tri(inv_sqrt_scale, p);
    
    lmarg_null[0] = lmvgammafn(0.5*(dn-1.0), p) - dp*log(2.0) - 0.5*dp*log(dn) - 0.5*dp*(dn-1.0)*log(pi) - 0.5*(dn-1.0)*(dp*log(dn-1.0) + log_det_var_y);
    
    //double BigConst = (lgammafn(0.5*df_imp*(dp+1.0)) + lgammafn(0.5*(df_imp+dp))
    //                   + 0.5*dp*log(dn)
    //                   - lmvgammafn(0.5*df_imp, p) - 2.0*lgammafn(0.5*df_imp)
    //                   - 0.5*(dp+2.0)*log_det_var_y
    //                   - 0.5*dp*log(df_imp) - 0.5*dp*log(pi)
    //                   - dp*log(scale_inflation));

    double BigConst = (lmvgammafn(df_imp, p) + lgammafn(0.5*(df_imp+dp))
                       + 0.5*dp*log(dn)
                       - 2.0*lmvgammafn(0.5*df_imp, p) - lgammafn(0.5*df_imp)
                       - 0.5*(dp+2.0)*log_det_scale_var
                       - 0.5*dp*log(df_imp) - 0.5*dp*log(pi)
                       - dp*log(scale_inflation));


    //double const2 = 0.5*dp*log(dn) + 0.5*(dn-1.0)*dp*log(dn-1.0) - 0.5*dp*log(2.0*pi) - 0.5*dp*(dn-1.0)*log(2.0) - lmvgammafn(0.5*(dn-1.0),p) - 0.5*(dp+2.0)*log_det_scale_var;
    
    GetRNGstate();
    int mu_position = 0, sig_position = 0, pp = p*p;
    double log_imp_samp_density = -9999.99, log_prior_density = -9999.99, null_loglik = -9999.9/*, log_null_post_density = -9999.99*/;
    double logdet_lam, tr_lamtlam, tt_term;
    
    for(s = 0; s < nmc; s++){
        //log_imp_samp_density = BigConst + imp_samp_mvF_mvT(mu_samp, inv_sqrt_sig_samp, p, n, df_imp, scale_inflation, &logdet_lam, &tr_lamtlam, &tt_term);
        log_imp_samp_density = BigConst + imp_samp_matF_mvT(mu_samp, inv_sqrt_sig_samp, p, n, df_imp, scale_inflation, &logdet_lam, &tr_lamtlam, &tt_term);

        log_prior_density = (dp+1.0)*log_det_tri(inv_sqrt_sig_samp, p) - dp*log(2.0);
        null_loglik = null_log_likelihood(mu_samp, inv_sqrt_sig_samp, n, p);
        //log_null_post_density = const2 + (dn+dp+1.0)*logdet_lam - 0.5*(dn-1.0)*tr_lamtlam - 0.5 * scale_inflation_sq * tt_term;
        
        
        rperm(perm, n, n);                         // randomly permute the data before Liu's sequential computing
        seqsamp_out = seqsamp_mv(mu_samp, inv_sqrt_sig_samp, alpha, omega1, omega2, n, p, nsub, perm, nclust_store + s);
        lwt_alt[s] = seqsamp_out + log_prior_density - log_imp_samp_density;
        lwt_null[s] = null_loglik + log_prior_density - log_imp_samp_density;
        //lwt_ratio[s] = seqsamp_out - null_loglik + log_null_post_density - log_imp_samp_density;
        sso[s] = seqsamp_out;
        limpd[s] = log_imp_samp_density;
        if(to_store){
            copy_vec2vec(mu_samp, mu_store + mu_position, p);
            mu_position += p;
            copy_mat2vec(inv_sqrt_sig_samp, inv_sqrt_sigma_store + sig_position, p, p, 1);
            sig_position += pp;
        }
    }
    PutRNGstate();
}


