#ifndef PROJECT_CC_HH
#define PROJECT_CC_HH

void project(const float rbin, const int numbins,
             const float* emiss, float* sb);

void add_sb_prof(const float rbin, const int nbins, const float *sb,
		 const float xc, const float yc,
		 const int xw, const int yw, float* img);

double logLikelihood(const int nelem, const float* data, const float* model);
float logLikelihoodSIMD(const int nelem, const float* data, const float* model);
float logLikelihoodSIMDMasked(const int nelem, const float* data,
			      const float* model, const int* mask);

void resamplePSF(int psf_nx, int psf_ny,
                 float psf_pixsize,
                 float psf_ox, float psf_oy,
                 const float *psf,
                 int oversample,
                 int img_nx, int img_ny,
                 float img_pixsize,
                 float *img);

void clipMin(float minval, int ny, int nx, float* arr);
void clipMax(float maxval, int ny, int nx, float* arr);

#endif
