#include <vector>
#include <limits>
#include <cstdio>

namespace
{
  constexpr int out_masked = -1;
  constexpr int out_unbinned = -2;

  template <class T> inline T sqr(T v)
  {
    return v*v;
  }

  template <class T> struct Point
  {
    Point() : x(0), y(0) {}
    Point(T _x, T _y) : x(_x), y(_y) {}
    T x, y;
  };

  template <class T> Point<T> operator+(const Point<T>& a, const Point<T> &b)
  {
    return Point<T>(a.x+b.x, a.y+b.y);
  }

  typedef Point<int> PointI;
  typedef Point<float> PointF;
  typedef std::vector<PointI> PointIVec;
  typedef std::vector<PointF> PointFVec;
  typedef std::vector<PointIVec> PointIVecVec;

  // collect pixels as a function of radius squared
  PointIVecVec make_circle_cache(int maxx, int maxy)
  {
    PointIVecVec rpts(maxx*maxx+maxy*maxy+1);
    for(int y=-maxy; y<=maxy; ++y)
      for(int x=-maxx; x<=maxx; ++x)
        rpts[x*x+y*y].push_back(PointI(x,y));

    return rpts;
  }

  // find nearest unbinned pixel to inpt
  PointI find_nearest(const PointIVecVec& circlecache,
                      int xw, int yw, const int* binimg, PointI inpt)
  {
    for(auto const &pv : circlecache)
      for(auto const &pt : pv)
        {
          PointI npt = inpt+pt;
          if(npt.x>=0 && npt.y>=0 && npt.x<xw && npt.y<yw &&
             binimg[npt.x+npt.y*xw]==out_unbinned)
            return npt;
        }
    return PointI(-1,-1);
  }

  // find highest pixel > mincts
  // if not found, then bin up and try again

  PointI find_initial_pixel(int xw, int yw, const float *inimg, const int *mask,
                            int bin, float mincts)
  {
    const int nx = xw/bin + (xw%bin != 0 ? 1 : 0);
    const int ny = yw/bin + (yw%bin != 0 ? 1 : 0);

    // binned too large
    if(nx==1 && ny==1)
      {
        return PointI(xw/2, yw/2);
      }

    float totmax = -1;
    PointI best(-1,-1);

    for(int yi=0; yi<ny; ++yi)
      for(int xi=0; xi<nx; ++xi)
        {
          float tot = 0;
          float tot_wx = 0;
          float tot_wy = 0;
          for(int yd=0; yd<bin; ++yd)
            for(int xd=0; xd<bin; ++xd)
              {
                int x = xi*bin+xd;
                int y = yi*bin+yd;
                int idx = x + y*xw;
                if(mask[idx])
                  {
                    tot += inimg[idx];
                    tot_wx += inimg[idx]*x;
                    tot_wy += inimg[idx]*y;
                  }
              }
          if(tot > totmax && tot > mincts)
            {
              totmax = tot;
              best = PointI(tot_wx*(1/tot), tot_wy*(1/tot));
            }
        }

    if(totmax > 0)
      return best;
    else
      return find_initial_pixel(xw, yw, inimg, mask, bin*2, mincts);
  }

  // compute centroids using inimg squared weighting
  PointIVec getMapCentroids(int xw, int yw, const float *inimg,
                            const int *binimg)
  {
    PointFVec pt_wt;
    std::vector<float> wts;

    for(int y=0; y<yw; ++y)
      for(int x=0; x<xw; ++x)
        {
          const int bin = binimg[x+y*xw];
          if(bin >= 0)
            {
              while(size_t(bin) >= wts.size())
                {
                  wts.push_back(0);
                  pt_wt.push_back(PointF());
                }
              const float wt = inimg[x+y*xw];
              wts[bin] += wt;
              pt_wt[bin] = pt_wt[bin] + PointF(wt*x, wt*y);
            }
        }

    PointIVec cen;
    for(size_t i=0; i<wts.size(); ++i)
      if(wts[i] > 0)
        {
          PointI c(pt_wt[i].x*(1/wts[i]), pt_wt[i].y*(1/wts[i]));
          cen.push_back(c);
        }
    return cen;
  }

}

void buildVoronoiMap_cc(int xw, int yw, const float *inimg,
                        int *binimg)
{
  const PointIVec centroids = getMapCentroids(xw, yw, inimg, binimg);
  const int num_cent = centroids.size();

  for(int y=0; y<yw; ++y)
    for(int x=0; x<xw; ++x)
      {
        if(binimg[x+y*xw] < 0)
          continue;

        // slow linear scan: make kd-tree later
        int near = -1;
        int near_r2 = std::numeric_limits<int>::max();
        for(int i=0; i<num_cent; ++i)
          {
            const int r2 = sqr(x-centroids[i].x)+sqr(y-centroids[i].y);
            near    = r2<near_r2 ? i  : near;
            near_r2 = r2<near_r2 ? r2 : near_r2;
          }
        binimg[x+y*xw] = near;
      }
}



void accreteBinImage_cc(int xw, int yw, const float *inimg, const int *mask,
			float threshcts, int *binimg)
{
  // clean output
  for(int i=0; i<xw*yw; ++i)
    {
      binimg[i] = mask[i] ? out_unbinned : out_masked;
    }

  // find an initial pixel (highest value with some minimum number of
  // counts)
  PointI startpix = find_initial_pixel(xw, yw, inimg, mask, 1, 10.f);

  // get pixel indices with some squared-radius
  const PointIVecVec circlecache = make_circle_cache(xw, yw);

  // keep track of total centroid and weighting factor
  PointF cen;
  float cen_wt = 0;

  for(int binidx=0; startpix.x>=0; ++binidx)
    {
      PointF thiscen;
      float thisweight = 0;
      PointI nextpix = startpix;

      while(thisweight < threshcts && nextpix.x >= 0)
	{
	  // S/N ~ cts/sqrt(cts) (ignoring background)
	  // density in C&C03 paper = (S/N)**2 = cts
	  const float cts = inimg[nextpix.x+nextpix.y*xw];
	  thisweight += cts;
	  thiscen.x += nextpix.x*cts;
	  thiscen.y += nextpix.y*cts;
	  binimg[nextpix.x+nextpix.y*xw] = binidx;

	  if(thisweight > 0)
	    {
	      startpix.x = int(thiscen.x*(1/thisweight));
	      startpix.y = int(thiscen.y*(1/thisweight));
	    }

	  nextpix = find_nearest(circlecache, xw, yw, binimg,
                                 startpix);
	}

      // no more pixels
      if(nextpix.x<0)
	  break;

      cen = cen + thiscen;
      cen_wt += thisweight;

      startpix = find_nearest(circlecache, xw, yw, binimg,
                              PointI(int(cen.x*(1/cen_wt)),
                                     int(cen.y*(1/cen_wt))));
    }
}
