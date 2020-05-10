#ifndef BINIMG_CC_HH
#define BINIMG_CC_HH

void buildVoronoiMap_cc(int xw, int yw, const float *inimg,
                        int *binimg);

void accreteBinImage_cc(int xw, int yw, const float *inimg, const int *maskimg,
                        float thresh, int *binimg);

#endif
