#include "mex.h"
#include "math.h"

mxArray* distmatsr(const mxArray* mat, const mxArray* sr, const mxArray* sl) {
	const double* pmat = mxGetPr(mat);
	const double* psr = mxGetPr(sr);
	const double* psl = mxGetPr(sl);
	
	int m = mxGetM(mat);
	int n = mxGetN(mat);
	int S = mxGetNumberOfElements(sl);
	
	mxArray* dist;
	double* pr;
	
	int i,j,k,p;
	double* pl = psl;
	double* pu = psr;

	dist = mxCreateDoubleMatrix(m, S, mxREAL);
	pr =  mxGetPr(dist);

	for (p = 0; p < S; p++) {
		int len = (int) (*(pl++));
		int s = n - len + 1;
		double sqrtN = sqrt(len);
		for (i = 0; i < m; i++) {
			double rMin = 1.79e+308;
			for (k = 0; k < s; k++) {
				double* pm = pmat + i + k*m;
				double* pv = pu;
				double r = 0.0;
				for (j = 0; j < len; j++) {
					double d = *pm - *pv;
					pm += m;
					pv++;
					r += d*d;
				}
				if (r < rMin) rMin = r;
			}
			*(pr++) = sqrt(rMin)/sqrtN;
		}
		pu += len;
	}
	return dist;
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	plhs[0] = distmatsr(prhs[0], prhs[1], prhs[2]);
}