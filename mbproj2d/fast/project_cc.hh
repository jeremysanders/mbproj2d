#ifndef PROJECT_CC_HH
#define PROJECT_CC_HH

void project(const float rbin, const int numbins,
             const float* emiss, float* sb);

void add_sb_prof(const float rbin, const int nbins, const float *sb,
		 const float xc, const float yc,
		 const int xw, const int yw, float* img);

double logLikelihood(const int nelem, const float* data, const float* model);
float logLikelihoodAVX(const int nelem, const float* data, const float* model);
float logLikelihoodAVXMasked(const int nelem, const float* data,
			     const float* model, const int* mask);

#endif
