#include "mex.h"
#include "math.h"

mxArray* distmatsr(const mxArray* mat, const mxArray* sr, const mxArray* sl, const mxArray* mcx, const mxArray* mcx2) {
	const double* pmat = mxGetPr(mat);
	const double* psr = mxGetPr(sr);
	const double* psl = mxGetPr(sl);
	const double* pcx = mxGetPr(mcx);
	const double* pcx2 = mxGetPr(mcx2);
	
	int m = mxGetM(mat);
	int n = mxGetN(mat);
	int S = mxGetNumberOfElements(sl);
	
	mxArray* dist;
	double* pr;
	
	int i,j,k,p,q;
	double* pl = psl;
	double* pu = psr;

	dist = mxCreateDoubleMatrix(m, S, mxREAL);
	pr =  mxGetPr(dist);
	
	for (p = 0; p < S; p++) {
		int len = (int) (*(pl++));
		double sqrtN = sqrt(len);
		double sumy = 0.0;
		double sumy2 = 0.0;
		for(q=0 ; q<len; q++) {
			sumy = sumy + *(pu+q);
			sumy2 = sumy2 + (*(pu+q))*(*(pu+q));
		}
		double meany = sumy/len;
		double sigmay = sqrt((sumy2/len) - meany*meany);
		for (i=0; i<m; i++) {
			double rMin = 1.79e+308;
			for (j=0; j<n-len+1; j++) {
				double sumxy = 0.0;
				for(k=0; k<len; k++) {
					sumxy += (*(pmat+i+(k+j)*m))*(*(pu+k));
				}
				double sumx = *(pcx+i+(j+len)*m) - *(pcx+i+j*m);
				double sumx2 = *(pcx2+i+(j+len)*m) - *(pcx2+i+j*m);
				double meanx = sumx/len;
				double sigmax = sqrt(sumx2/len-meanx*meanx);
				double d = (1-(sumxy/len - meanx*meany)/(sigmax*sigmay))*2*len;
				d = d<0? 0: d;
				if (d < rMin) rMin = d;
			}
			*(pr++) = sqrt(rMin)/sqrtN;
		}
		pu += len;
	}
	return dist;
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	plhs[0] = distmatsr(prhs[0], prhs[1], prhs[2], prhs[3], prhs[4]);
}