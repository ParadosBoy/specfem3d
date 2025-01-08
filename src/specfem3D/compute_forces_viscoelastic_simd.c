#include <stdio.h>
#include <arm_neon.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
void dumm_loc_simd(float *displ_p,float eta,float *veloc_p,float *dummyx_loc_p,float *dummyy_loc_p,float *dummyz_loc_p)
{
/*	dummyx_loc_p[0]=displ_p[0]+eta*veloc_p[0];
	dummyy_loc_p[0]=displ_p[1]+eta*veloc_p[1];
	dummyz_loc_p[0]=displ_p[2]+eta*veloc_p[2];*/
	float32x4_t displ = vld1q_f32(displ_p);
    	float32x4_t veloc = vld1q_f32(veloc_p);
	
	float32x4_t eta_vec = vdupq_n_f32(eta);
	float32x4_t dumm = vaddq_f32(displ,vmulq_f32(eta_vec, veloc));

	dummyx_loc_p[0] = vgetq_lane_f32(dumm, 0);
    	dummyy_loc_p[0] = vgetq_lane_f32(dumm, 1);
    	dummyz_loc_p[0] = vgetq_lane_f32(dumm, 2);
}
void dumm_loc_simd_2(float eta,float *veloc_p,float *dummyx_loc_p,float *dummyy_loc_p,float *dummyz_loc_p)
{
/*      dummyx_loc_p[0]=eta*veloc_p[0];
        dummyy_loc_p[0]=eta*veloc_p[1];
        dummyz_loc_p[0]=eta*veloc_p[2];*/
        float32x4_t veloc = vld1q_f32(veloc_p);

        float32x4_t eta_vec = vdupq_n_f32(eta);
        float32x4_t dumm = vmulq_f32(eta_vec, veloc);

        dummyx_loc_p[0] = vgetq_lane_f32(dumm, 0);
        dummyy_loc_p[0] = vgetq_lane_f32(dumm, 1);
        dummyz_loc_p[0] = vgetq_lane_f32(dumm, 2);
}

void temp_simd(float *dummyx_loc_p,float *dummyy_loc_p,float *dummyz_loc_p,float hp,float tempx,float tempy,float tempz)
{
	tempx+=dummyx_loc_p[0]*hp;
	tempy+=dummyy_loc_p[0]*hp;
	tempz+=dummyz_loc_p[0]*hp;

}
void du_simd(float *xixl,float *xiyl,float *xizl,float *etaxl,float *etayl,float *etazl,float *gammaxl,float *gammayl,float *gammazl,float *tempx1,float *tempx2,float *tempx3,float *duxdxl,float *duxdyl,float *duxdzl,float *tempy1,float *tempy2,float *tempy3,float *duydxl,float *duydyl,float *duydzl,float *tempz1,float *tempz2,float *tempz3,float *duzdxl,float *duzdyl,float *duzdzl)
{
	for(int i=0;i<124;i+=4){
	  float32x4_t xixl_vec = vld1q_f32(&xixl[i]);
       	  float32x4_t xiyl_vec = vld1q_f32(&xiyl[i]);	
	  float32x4_t xizl_vec = vld1q_f32(&xizl[i]);
	  float32x4_t etaxl_vec = vld1q_f32(&etaxl[i]);
	  float32x4_t etayl_vec = vld1q_f32(&etayl[i]);
	  float32x4_t etazl_vec = vld1q_f32(&etazl[i]);
	  float32x4_t gammaxl_vec = vld1q_f32(&gammaxl[i]);
	  float32x4_t gammayl_vec = vld1q_f32(&gammayl[i]);
	  float32x4_t gammazl_vec = vld1q_f32(&gammazl[i]);
	  float32x4_t tempx1_vec = vld1q_f32(&tempx1[i]);
	  float32x4_t tempx2_vec = vld1q_f32(&tempx2[i]);
	  float32x4_t tempx3_vec = vld1q_f32(&tempx3[i]);
	  float32x4_t tempy1_vec = vld1q_f32(&tempy1[i]);
          float32x4_t tempy2_vec = vld1q_f32(&tempy2[i]);
          float32x4_t tempy3_vec = vld1q_f32(&tempy3[i]);
	  float32x4_t tempz1_vec = vld1q_f32(&tempz1[i]);
          float32x4_t tempz2_vec = vld1q_f32(&tempz2[i]);
          float32x4_t tempz3_vec = vld1q_f32(&tempz3[i]);

	  float32x4_t duxdxl_vec=vmulq_f32(xixl_vec, tempx1_vec);
	  duxdxl_vec=vmlaq_f32(duxdxl_vec, etaxl_vec, tempx2_vec);
	  duxdxl_vec=vmlaq_f32(duxdxl_vec, gammaxl_vec, tempx3_vec);
	  vst1q_f32(&duxdxl[i], duxdxl_vec);
	
	  float32x4_t duxdyl_vec=vmulq_f32(xiyl_vec, tempx1_vec);
          duxdyl_vec=vmlaq_f32(duxdyl_vec, etayl_vec, tempx2_vec);
          duxdyl_vec=vmlaq_f32(duxdyl_vec, gammayl_vec, tempx3_vec);
          vst1q_f32(&duxdyl[i], duxdyl_vec);

	  float32x4_t duxdzl_vec=vmulq_f32(xizl_vec, tempx1_vec);
          duxdzl_vec=vmlaq_f32(duxdzl_vec, etazl_vec, tempx2_vec);
          duxdzl_vec=vmlaq_f32(duxdzl_vec, gammazl_vec, tempx3_vec);
          vst1q_f32(&duxdzl[i], duxdzl_vec);

	  float32x4_t duydxl_vec=vmulq_f32(xixl_vec, tempy1_vec);
          duydxl_vec=vmlaq_f32(duydxl_vec, etaxl_vec, tempy2_vec);
          duydxl_vec=vmlaq_f32(duydxl_vec, gammaxl_vec, tempy3_vec);
          vst1q_f32(&duydxl[i], duydxl_vec);

          float32x4_t duydyl_vec=vmulq_f32(xiyl_vec, tempy1_vec);
          duydyl_vec=vmlaq_f32(duydyl_vec, etayl_vec, tempy2_vec);
          duydyl_vec=vmlaq_f32(duydyl_vec, gammayl_vec, tempy3_vec);
          vst1q_f32(&duydyl[i], duydyl_vec);

          float32x4_t duydzl_vec=vmulq_f32(xizl_vec, tempy1_vec);
          duydzl_vec=vmlaq_f32(duydzl_vec, etazl_vec, tempy2_vec);
          duydzl_vec=vmlaq_f32(duydzl_vec, gammazl_vec, tempy3_vec);
          vst1q_f32(&duydzl[i], duydzl_vec);

	  float32x4_t duzdxl_vec=vmulq_f32(xixl_vec, tempz1_vec);
          duzdxl_vec=vmlaq_f32(duzdxl_vec, etaxl_vec, tempz2_vec);
          duzdxl_vec=vmlaq_f32(duzdxl_vec, gammaxl_vec, tempz3_vec);
          vst1q_f32(&duzdxl[i], duzdxl_vec);

          float32x4_t duzdyl_vec=vmulq_f32(xiyl_vec, tempz1_vec);
          duzdyl_vec=vmlaq_f32(duzdyl_vec, etayl_vec, tempz2_vec);
          duzdyl_vec=vmlaq_f32(duzdyl_vec, gammayl_vec, tempz3_vec);
          vst1q_f32(&duzdyl[i], duzdyl_vec);

          float32x4_t duzdzl_vec=vmulq_f32(xizl_vec, tempz1_vec);
          duzdzl_vec=vmlaq_f32(duzdzl_vec, etazl_vec, tempz2_vec);
          duzdzl_vec=vmlaq_f32(duzdzl_vec, gammazl_vec, tempz3_vec);
          vst1q_f32(&duzdzl[i], duzdzl_vec);
	}
	duxdxl[124] = xixl[124]*tempx1[124] + etaxl[124]*tempx2[124] + gammaxl[124]*tempx3[124];
        duxdyl[124] = xiyl[124]*tempx1[124] + etayl[124]*tempx2[124] + gammayl[124]*tempx3[124];
        duxdzl[124] = xizl[124]*tempx1[124] + etazl[124]*tempx2[124] + gammazl[124]*tempx3[124];

        duydxl[124] = xixl[124]*tempy1[124] + etaxl[124]*tempy2[124] + gammaxl[124]*tempy3[124];
        duydyl[124] = xiyl[124]*tempy1[124] + etayl[124]*tempy2[124] + gammayl[124]*tempy3[124];
        duydzl[124] = xizl[124]*tempy1[124] + etazl[124]*tempy2[124] + gammazl[124]*tempy3[124];

        duzdxl[124] = xixl[124]*tempz1[124] + etaxl[124]*tempz2[124] + gammaxl[124]*tempz3[124];
        duzdyl[124] = xiyl[124]*tempz1[124] + etayl[124]*tempz2[124] + gammayl[124]*tempz3[124];
        duzdzl[124] = xizl[124]*tempz1[124] + etazl[124]*tempz2[124] + gammazl[124]*tempz3[124];
}
void stresses_simd(
    float* c11, float* c12, float* c13, float* c14, float* c15, float* c16,
    float* c22, float* c23, float* c24, float* c25, float* c26,
    float* c33, float* c34, float* c35, float* c36,
    float* c44, float* c45, float* c46,
    float* c55, float* c56, float* c66,
    float* duxdyl, float* duydxl, float* duzdxl, float* duxdzl,
    float* duzdyl, float* duydzl, float* duxdxl, float* duydyl, float* duzdzl,
    float* kappal, float* mul, float FOUR_THIRDS,
    float* R_xx_sum, float* R_yy_sum, float* R_trace_kappa_sum,
    float* R_xy_sum, float* R_xz_sum, float* R_yz_sum,
    bool ATTENUATION, bool ANISOTROPY,bool is_CPML, bool MOVIE_VOLUME_STRESS, 
    float* stress_xx, float* stress_yy, float* stress_zz, float* stress_xy, float* stress_xz, float* stress_yz,
    int32_t ispec_irreg,
    float* xixl, float* xiyl, float* xizl,
    float* etaxl, float* etayl, float* etazl,
    float* gammaxl, float* gammayl, float* gammazl,
    float* jacobianl,
    float* tempx1, float* tempx2, float* tempx3,
    float* tempy1, float* tempy2, float* tempy3,
    float* tempz1, float* tempz2, float* tempz3,
    float jacobian_regular, float xix_regular
)
{
	float32x4_t tempx1_vec,tempy1_vec,tempz1_vec,tempx2_vec,tempy2_vec,tempz2_vec,tempx3_vec,tempy3_vec,tempz3_vec;
	float32x4_t sigma_xx, sigma_yy,sigma_zz,sigma_xy,sigma_xz,sigma_yz;
	for(int i=0;i<124;i+=4){
	  float32x4_t duxdxl_vec = vld1q_f32(&duxdxl[i]);
	  float32x4_t duxdyl_vec = vld1q_f32(&duxdyl[i]);
	  float32x4_t duydxl_vec = vld1q_f32(&duydxl[i]);
	  float32x4_t duydyl_vec = vld1q_f32(&duydyl[i]);
	  float32x4_t duzdxl_vec = vld1q_f32(&duzdxl[i]);
	  float32x4_t duxdzl_vec = vld1q_f32(&duxdzl[i]);
	  float32x4_t duzdyl_vec = vld1q_f32(&duzdyl[i]);
	  float32x4_t duydzl_vec = vld1q_f32(&duydzl[i]);
	  float32x4_t duzdzl_vec = vld1q_f32(&duzdzl[i]);

	  float32x4_t duxdyl_plus_duydxl_vec = vaddq_f32(duxdyl_vec,duydxl_vec);
          float32x4_t duzdxl_plus_duxdzl_vec = vaddq_f32(duzdxl_vec,duxdzl_vec);
          float32x4_t duzdyl_plus_duydzl_vec = vaddq_f32(duzdyl_vec,duydzl_vec);
	  if(ANISOTROPY){
	    float32x4_t c11_vec = vld1q_f32(&c11[i]);
	    float32x4_t c12_vec = vld1q_f32(&c12[i]);
	    float32x4_t c13_vec = vld1q_f32(&c13[i]);
	    float32x4_t c14_vec = vld1q_f32(&c14[i]);
	    float32x4_t c15_vec = vld1q_f32(&c15[i]);
	    float32x4_t c16_vec = vld1q_f32(&c16[i]);
	    float32x4_t c22_vec = vld1q_f32(&c22[i]);
	    float32x4_t c23_vec = vld1q_f32(&c23[i]);
	    float32x4_t c24_vec = vld1q_f32(&c24[i]);
	    float32x4_t c25_vec = vld1q_f32(&c25[i]);
	    float32x4_t c26_vec = vld1q_f32(&c26[i]);
	    float32x4_t c33_vec = vld1q_f32(&c33[i]);
	    float32x4_t c34_vec = vld1q_f32(&c34[i]);
	    float32x4_t c35_vec = vld1q_f32(&c35[i]);
	    float32x4_t c36_vec = vld1q_f32(&c36[i]);
	    float32x4_t c44_vec = vld1q_f32(&c44[i]);
	    float32x4_t c45_vec = vld1q_f32(&c45[i]);
	    float32x4_t c46_vec = vld1q_f32(&c46[i]);
	    float32x4_t c55_vec = vld1q_f32(&c55[i]);
	    float32x4_t c56_vec = vld1q_f32(&c56[i]);
	    float32x4_t c66_vec = vld1q_f32(&c66[i]);
	  
	    sigma_xx = vmulq_f32(c11_vec, duxdxl_vec);
	    sigma_xx = vfmaq_f32(sigma_xx,c16_vec,duxdyl_plus_duydxl_vec);
	    sigma_xx = vfmaq_f32(sigma_xx,c12_vec,duydyl_vec);
	    sigma_xx = vfmaq_f32(sigma_xx,c15_vec,duzdxl_plus_duxdzl_vec);
	    sigma_xx = vfmaq_f32(sigma_xx,c14_vec,duzdyl_plus_duydzl_vec);
	    sigma_xx = vfmaq_f32(sigma_xx,c13_vec,duzdzl_vec);
	    sigma_yy = vmulq_f32(c12_vec, duxdxl_vec);
            sigma_yy = vfmaq_f32(sigma_yy,c26_vec,duxdyl_plus_duydxl_vec);
            sigma_yy = vfmaq_f32(sigma_yy,c22_vec,duydyl_vec);
            sigma_yy = vfmaq_f32(sigma_yy,c25_vec,duzdxl_plus_duxdzl_vec);
            sigma_yy = vfmaq_f32(sigma_yy,c24_vec,duzdyl_plus_duydzl_vec);
            sigma_yy = vfmaq_f32(sigma_yy,c23_vec,duzdzl_vec);

	    sigma_zz = vmulq_f32(c13_vec, duxdxl_vec);
            sigma_zz = vfmaq_f32(sigma_zz,c36_vec,duxdyl_plus_duydxl_vec);
            sigma_zz = vfmaq_f32(sigma_zz,c23_vec,duydyl_vec);
            sigma_zz = vfmaq_f32(sigma_zz,c35_vec,duzdxl_plus_duxdzl_vec);
            sigma_zz = vfmaq_f32(sigma_zz,c34_vec,duzdyl_plus_duydzl_vec);
            sigma_zz = vfmaq_f32(sigma_zz,c33_vec,duzdzl_vec);

	    sigma_xy = vmulq_f32(c16_vec, duxdxl_vec);
            sigma_xy = vfmaq_f32(sigma_xy,c66_vec,duxdyl_plus_duydxl_vec);
            sigma_xy = vfmaq_f32(sigma_xy,c26_vec,duydyl_vec);
            sigma_xy = vfmaq_f32(sigma_xy,c56_vec,duzdxl_plus_duxdzl_vec);
            sigma_xy = vfmaq_f32(sigma_xy,c46_vec,duzdyl_plus_duydzl_vec);
            sigma_xy = vfmaq_f32(sigma_xy,c36_vec,duzdzl_vec);
	    
	    sigma_xz = vmulq_f32(c15_vec, duxdxl_vec);
            sigma_xz = vfmaq_f32(sigma_xz,c56_vec,duxdyl_plus_duydxl_vec);
            sigma_xz = vfmaq_f32(sigma_xz,c25_vec,duydyl_vec);
            sigma_xz = vfmaq_f32(sigma_xz,c55_vec,duzdxl_plus_duxdzl_vec);
            sigma_xz = vfmaq_f32(sigma_xz,c45_vec,duzdyl_plus_duydzl_vec);
            sigma_xz = vfmaq_f32(sigma_xz,c35_vec,duzdzl_vec);

	    sigma_yz = vmulq_f32(c14_vec, duxdxl_vec);
            sigma_yz = vfmaq_f32(sigma_yz,c46_vec,duxdyl_plus_duydxl_vec);
            sigma_yz = vfmaq_f32(sigma_yz,c24_vec,duydyl_vec);
            sigma_yz = vfmaq_f32(sigma_yz,c45_vec,duzdxl_plus_duxdzl_vec);
            sigma_yz = vfmaq_f32(sigma_yz,c44_vec,duzdyl_plus_duydzl_vec);
            sigma_yz = vfmaq_f32(sigma_yz,c34_vec,duzdzl_vec);
	  }
	  else{ 
	    float32x4_t kappal_vec = vld1q_f32(&kappal[i]);
	    float32x4_t mul_vec = vld1q_f32(&mul[i]);
	    float32x4_t FOUR_THIRDS_vec = vdupq_n_f32(FOUR_THIRDS);
	    float32x4_t vec = vdupq_n_f32(2.0f);
	    float32x4_t lambdalplus2mul_vec = vaddq_f32(kappal_vec,vmulq_f32(FOUR_THIRDS_vec, mul_vec));
	    float32x4_t lambdal_vec = vsubq_f32(lambdalplus2mul_vec,vmulq_f32(vec, mul_vec));

	    sigma_xx = vmulq_f32(lambdalplus2mul_vec, duxdxl_vec);
	    sigma_xx = vfmaq_f32(sigma_xx,lambdal_vec,vaddq_f32(duydyl_vec,duzdzl_vec));

	    sigma_yy = vmulq_f32(lambdalplus2mul_vec, duydyl_vec);
            sigma_yy = vfmaq_f32(sigma_yy,lambdal_vec,vaddq_f32(duxdxl_vec,duzdzl_vec));
            
	    sigma_zz = vmulq_f32(lambdalplus2mul_vec, duzdzl_vec);
            sigma_zz = vfmaq_f32(sigma_zz,lambdal_vec,vaddq_f32(duxdxl_vec,duydyl_vec));

	    sigma_xy = vmulq_f32(mul_vec,duxdyl_plus_duydxl_vec);
	    sigma_xz = vmulq_f32(mul_vec,duzdxl_plus_duxdzl_vec);
	    sigma_yz = vmulq_f32(mul_vec,duzdyl_plus_duydzl_vec);
	  }
	  if (MOVIE_VOLUME_STRESS){
	    vst1q_f32(&stress_xx[i], sigma_xx);
            vst1q_f32(&stress_yy[i], sigma_yy);
            vst1q_f32(&stress_zz[i], sigma_zz);

            vst1q_f32(&stress_xy[i], sigma_xy);
            vst1q_f32(&stress_xz[i], sigma_xz);
            vst1q_f32(&stress_yz[i], sigma_yz);
	  }
	  float32x4_t R_xx_sum_vec = vld1q_f32(&R_xx_sum[i]);
	  float32x4_t R_yy_sum_vec = vld1q_f32(&R_yy_sum[i]);
	  float32x4_t R_xy_sum_vec = vld1q_f32(&R_xy_sum[i]);
	  float32x4_t R_xz_sum_vec = vld1q_f32(&R_xz_sum[i]);
	  float32x4_t R_yz_sum_vec = vld1q_f32(&R_yz_sum[i]);
	  float32x4_t R_trace_kappa_sum_vec = vld1q_f32(&R_trace_kappa_sum[i]);
	  if ((ATTENUATION) && (!is_CPML)){
            sigma_xx = vsubq_f32(sigma_xx,R_xx_sum_vec);
	    sigma_xx = vsubq_f32(sigma_xx,R_trace_kappa_sum_vec);
	    sigma_yy = vsubq_f32(sigma_yy,R_yy_sum_vec);
            sigma_yy = vsubq_f32(sigma_yy,R_trace_kappa_sum_vec);
		
	    sigma_zz = vaddq_f32(sigma_zz,R_xx_sum_vec);
	    sigma_zz = vaddq_f32(sigma_zz,R_yy_sum_vec);
	    sigma_zz = vsubq_f32(sigma_zz,R_trace_kappa_sum_vec);

	    sigma_xy = vsubq_f32(sigma_xy,R_xy_sum_vec);
	    sigma_xz = vsubq_f32(sigma_xz,R_xz_sum_vec);
	    sigma_yz = vsubq_f32(sigma_yz,R_yz_sum_vec);
	  }
  	  if (!is_CPML){
	    if (ispec_irreg != 0){
	      float32x4_t xixl_vec = vld1q_f32(&xixl[i]);
	      float32x4_t xiyl_vec = vld1q_f32(&xiyl[i]);
	      float32x4_t xizl_vec = vld1q_f32(&xizl[i]);

	      float32x4_t etaxl_vec = vld1q_f32(&etaxl[i]);
	      float32x4_t etayl_vec = vld1q_f32(&etayl[i]);
	      float32x4_t etazl_vec = vld1q_f32(&etazl[i]);

	      float32x4_t gammaxl_vec = vld1q_f32(&gammaxl[i]);
	      float32x4_t gammayl_vec = vld1q_f32(&gammayl[i]);
	      float32x4_t gammazl_vec = vld1q_f32(&gammazl[i]);

	      float32x4_t jacobianl_vec = vld1q_f32(&jacobianl[i]);
	      tempx1_vec = vmulq_f32(sigma_xx, xixl_vec);
	      tempx1_vec = vfmaq_f32(tempx1_vec,sigma_xy,xiyl_vec);
	      tempx1_vec = vfmaq_f32(tempx1_vec,sigma_xz,xizl_vec);
	      tempx1_vec = vmulq_f32(tempx1_vec, jacobianl_vec);
	      tempy1_vec = vmulq_f32(sigma_xy, xixl_vec);
              tempy1_vec = vfmaq_f32(tempy1_vec,sigma_yy,xiyl_vec);
              tempy1_vec = vfmaq_f32(tempy1_vec,sigma_yz,xizl_vec);
              tempy1_vec = vmulq_f32(tempy1_vec, jacobianl_vec);

	      tempz1_vec = vmulq_f32(sigma_xz, xixl_vec);
              tempz1_vec = vfmaq_f32(tempz1_vec,sigma_yz,xiyl_vec);
              tempz1_vec = vfmaq_f32(tempz1_vec,sigma_zz,xizl_vec);
              tempz1_vec = vmulq_f32(tempz1_vec, jacobianl_vec);

	      tempx2_vec = vmulq_f32(sigma_xx, etaxl_vec);
              tempx2_vec = vfmaq_f32(tempx2_vec,sigma_xy,etayl_vec);
              tempx2_vec = vfmaq_f32(tempx2_vec,sigma_xz,etazl_vec);
              tempx2_vec = vmulq_f32(tempx2_vec, jacobianl_vec);

              tempy2_vec = vmulq_f32(sigma_xy, etaxl_vec);
              tempy2_vec = vfmaq_f32(tempy2_vec,sigma_yy,etayl_vec);
              tempy2_vec = vfmaq_f32(tempy2_vec,sigma_yz,etazl_vec);
              tempy2_vec = vmulq_f32(tempy2_vec, jacobianl_vec);

              tempz2_vec = vmulq_f32(sigma_xz, etaxl_vec);
              tempz2_vec = vfmaq_f32(tempz2_vec,sigma_yz,etayl_vec);
              tempz2_vec = vfmaq_f32(tempz2_vec,sigma_zz,etazl_vec);
              tempz2_vec = vmulq_f32(tempz2_vec, jacobianl_vec);
			
	      tempx3_vec = vmulq_f32(sigma_xx, gammaxl_vec);
              tempx3_vec = vfmaq_f32(tempx3_vec,sigma_xy,gammayl_vec);
              tempx3_vec = vfmaq_f32(tempx3_vec,sigma_xz,gammazl_vec);
              tempx3_vec = vmulq_f32(tempx3_vec, jacobianl_vec);

              tempy3_vec = vmulq_f32(sigma_xy, gammaxl_vec);
              tempy3_vec = vfmaq_f32(tempy3_vec,sigma_yy,gammayl_vec);
              tempy3_vec = vfmaq_f32(tempy3_vec,sigma_yz,gammazl_vec);
              tempy3_vec = vmulq_f32(tempy3_vec, jacobianl_vec);

              tempz3_vec = vmulq_f32(sigma_xz, gammaxl_vec);
              tempz3_vec = vfmaq_f32(tempz3_vec,sigma_yz,gammayl_vec);
              tempz3_vec = vfmaq_f32(tempz3_vec,sigma_zz,gammazl_vec);
              tempz3_vec = vmulq_f32(tempz3_vec, jacobianl_vec);
	     }
	    else{
	      float32x4_t jacobian_regular_vec = vdupq_n_f32(jacobian_regular);
	      float32x4_t xix_regular_vec = vdupq_n_f32(xix_regular);
	      
	      tempx1_vec = vmulq_f32(jacobian_regular_vec, sigma_xx);
	      tempx1_vec = vmulq_f32(tempx1_vec,xix_regular_vec);

	      tempy1_vec = vmulq_f32(jacobian_regular_vec, sigma_xy);
              tempy1_vec = vmulq_f32(tempy1_vec,xix_regular_vec);
	    
	      tempz1_vec = vmulq_f32(jacobian_regular_vec, sigma_xz);
              tempz1_vec = vmulq_f32(tempz1_vec,xix_regular_vec);

	      tempx2_vec = vmulq_f32(jacobian_regular_vec, sigma_xy);
              tempx2_vec = vmulq_f32(tempx2_vec,xix_regular_vec);

	      tempy2_vec = vmulq_f32(jacobian_regular_vec, sigma_yy);
              tempy2_vec = vmulq_f32(tempy2_vec,xix_regular_vec);

	      tempz2_vec = vmulq_f32(jacobian_regular_vec, sigma_yz);
              tempz2_vec = vmulq_f32(tempz2_vec,xix_regular_vec);

	      tempx3_vec = vmulq_f32(jacobian_regular_vec, sigma_xz);
              tempx3_vec = vmulq_f32(tempx3_vec,xix_regular_vec);

	      tempy3_vec = vmulq_f32(jacobian_regular_vec, sigma_yz);
              tempy3_vec = vmulq_f32(tempy3_vec,xix_regular_vec);

	      tempz3_vec = vmulq_f32(jacobian_regular_vec, sigma_zz);
              tempz3_vec = vmulq_f32(tempz3_vec,xix_regular_vec);
	      }
	}
	    vst1q_f32(&tempx1[i], tempx1_vec);
	    vst1q_f32(&tempy1[i], tempy1_vec);
	    vst1q_f32(&tempz1[i], tempz1_vec);

	    vst1q_f32(&tempx2[i], tempx2_vec);
	    vst1q_f32(&tempy2[i], tempy2_vec);
	    vst1q_f32(&tempz2[i], tempz2_vec);

	    vst1q_f32(&tempx3[i], tempx3_vec);
            vst1q_f32(&tempy3[i], tempy3_vec);
            vst1q_f32(&tempz3[i], tempz3_vec);
	}
	float sigma_xx_s,sigma_yy_s,sigma_zz_s,sigma_xy_s,sigma_xz_s,sigma_yz_s;
	float duxdyl_plus_duydxl = duxdyl[124] + duydxl[124];
	float duzdxl_plus_duxdzl = duzdxl[124] + duxdzl[124];
	float duzdyl_plus_duydzl = duzdyl[124] + duydzl[124];
	if (ANISOTROPY)
	{
          sigma_xx_s = c11[124] * duxdxl[124] + c16[124] * duxdyl_plus_duydxl + c12[124] * duydyl[124] + c15[124] * duzdxl_plus_duxdzl + c14[124] * duzdyl_plus_duydzl + c13[124] * duzdzl[124];
	  sigma_yy_s = c12[124] * duxdxl[124] + c26[124] * duxdyl_plus_duydxl + c22[124] * duydyl[124] + c25[124] * duzdxl_plus_duxdzl + c24[124] * duzdyl_plus_duydzl + c23[124] * duzdzl[124];
	  sigma_zz_s = c13[124] * duxdxl[124] + c36[124] * duxdyl_plus_duydxl + c23[124] * duydyl[124] + c35[124] * duzdxl_plus_duxdzl + c34[124] * duzdyl_plus_duydzl + c33[124] * duzdzl[124];
	  sigma_xy_s = c16[124] * duxdxl[124] + c66[124] * duxdyl_plus_duydxl + c26[124] * duydyl[124] + c56[124] * duzdxl_plus_duxdzl + c46[124] * duzdyl_plus_duydzl + c36[124] * duzdzl[124];
	  sigma_xz_s = c15[124] * duxdxl[124] + c56[124] * duxdyl_plus_duydxl + c25[124] * duydyl[124] + c55[124] * duzdxl_plus_duxdzl + c45[124] * duzdyl_plus_duydzl + c35[124] * duzdzl[124];
	  sigma_yz_s = c14[124] * duxdxl[124] + c46[124] * duxdyl_plus_duydxl + c24[124] * duydyl[124] + c45[124] * duzdxl_plus_duxdzl + c44[124] * duzdyl_plus_duydzl + c34[124] * duzdzl[124];
	}
	else{
	  float lambdalplus2mul = kappal[124] + FOUR_THIRDS * mul[124];
	  float lambdal = lambdalplus2mul - 2.0 * mul[124];
	  sigma_xx_s = lambdalplus2mul * duxdxl[124] + lambdal * (duydyl[124] + duzdzl[124]);
          sigma_yy_s = lambdalplus2mul * duydyl[124] + lambdal * (duxdxl[124] + duzdzl[124]);
          sigma_zz_s = lambdalplus2mul * duzdzl[124] + lambdal * (duxdxl[124] + duydyl[124]);

          sigma_xy_s = mul[124] * duxdyl_plus_duydxl;
          sigma_xz_s = mul[124] * duzdxl_plus_duxdzl;
          sigma_yz_s = mul[124] * duzdyl_plus_duydzl;
	  }
	if (MOVIE_VOLUME_STRESS){
            stress_xx[124] = sigma_xx_s;
            stress_yy[124] = sigma_yy_s;
            stress_zz[124] = sigma_zz_s;

            stress_xy[124] = sigma_xy_s;
            stress_xz[124] = sigma_xz_s;
            stress_yz[124] = sigma_yz_s;
          }
        if ((ATTENUATION) && (!is_CPML)) {
	  sigma_xx_s = sigma_xx_s - R_xx_sum[124] - R_trace_kappa_sum[124];
	  sigma_yy_s = sigma_yy_s - R_yy_sum[124] - R_trace_kappa_sum[124];
	  sigma_zz_s = sigma_zz_s + R_xx_sum[124] + R_yy_sum[124] - R_trace_kappa_sum[124];
	  sigma_xy_s = sigma_xy_s - R_xy_sum[124];
	  sigma_xz_s = sigma_xz_s - R_xz_sum[124];
	  sigma_yz_s = sigma_yz_s - R_yz_sum[124];
	}
	if (!is_CPML ){
            if (ispec_irreg != 0){
		tempx1[124] = jacobianl[124] * (sigma_xx_s * xixl[124] + sigma_xy_s * xiyl[124] + sigma_xz_s * xizl[124]);
                tempy1[124] = jacobianl[124] * (sigma_xy_s * xixl[124] + sigma_yy_s * xiyl[124] + sigma_yz_s * xizl[124]);
                tempz1[124] = jacobianl[124] * (sigma_xz_s * xixl[124] + sigma_yz_s * xiyl[124] + sigma_zz_s * xizl[124]);

                tempx2[124] = jacobianl[124] * (sigma_xx_s * etaxl[124] + sigma_xy_s * etayl[124] + sigma_xz_s * etazl[124]);
                tempy2[124] = jacobianl[124] * (sigma_xy_s * etaxl[124] + sigma_yy_s * etayl[124] + sigma_yz_s * etazl[124]);
                tempz2[124] = jacobianl[124] * (sigma_xz_s * etaxl[124] + sigma_yz_s * etayl[124] + sigma_zz_s * etazl[124]);

                tempx3[124] = jacobianl[124] * (sigma_xx_s * gammaxl[124] + sigma_xy_s * gammayl[124] + sigma_xz_s * gammazl[124]);
                tempy3[124] = jacobianl[124] * (sigma_xy_s * gammaxl[124] + sigma_yy_s * gammayl[124] + sigma_yz_s * gammazl[124]);
                tempz3[124] = jacobianl[124] * (sigma_xz_s * gammaxl[124] + sigma_yz_s * gammayl[124] + sigma_zz_s * gammazl[124]);
	   }
	    else{
	    	tempx1[124] = jacobian_regular * sigma_xx_s * xix_regular; 
                tempy1[124] = jacobian_regular * sigma_xy_s * xix_regular;
                tempz1[124] = jacobian_regular * sigma_xz_s * xix_regular; 

                tempx2[124] = jacobian_regular * sigma_xy_s * xix_regular;
                tempy2[124] = jacobian_regular * sigma_yy_s * xix_regular; 
                tempz2[124] = jacobian_regular * sigma_yz_s * xix_regular; 

          	tempx3[124] = jacobian_regular * sigma_xz_s * xix_regular;
                tempy3[124] = jacobian_regular * sigma_yz_s * xix_regular;
          	tempz3[124] = jacobian_regular * sigma_zz_s * xix_regular;
	    }
	}
}

void mxm5_3comp_singleA_simd(float* A,int n1,float* B1,float* B2,float* B3,float* C1,float* C2,float* C3,int n3)
{
	int i = 0;
	float32x4_t A_vec, A1_vec, A2_vec, A3_vec, A4_vec, A5_vec, B1_vec, B2_vec, B3_vec;
	float32x4_t C1_vec, C2_vec, C3_vec;
	A1_vec = vld1q_f32(&A[0]);
	A2_vec = vld1q_f32(&A[5]);
	A3_vec = vld1q_f32(&A[10]);
	A4_vec = vld1q_f32(&A[15]);
	A5_vec = vld1q_f32(&A[20]);
	for(int j = 0; j < n3; j++){
		i = 5 * j;
		B1_vec = vdupq_n_f32(B1[i]);
		B2_vec = vdupq_n_f32(B2[i]);
		B3_vec = vdupq_n_f32(B3[i]);

		C1_vec = vmulq_f32(A1_vec,B1_vec);
		C2_vec = vmulq_f32(A1_vec,B2_vec);
		C3_vec = vmulq_f32(A1_vec,B3_vec);

		B1_vec = vdupq_n_f32(B1[i+1]);
                B2_vec = vdupq_n_f32(B2[i+1]);
                B3_vec = vdupq_n_f32(B3[i+1]);

		C1_vec = vfmaq_f32(C1_vec,A2_vec,B1_vec);
                C2_vec = vfmaq_f32(C2_vec,A2_vec,B2_vec);
                C3_vec = vfmaq_f32(C3_vec,A2_vec,B3_vec);

		B1_vec = vdupq_n_f32(B1[i+2]);
                B2_vec = vdupq_n_f32(B2[i+2]);
                B3_vec = vdupq_n_f32(B3[i+2]);

                C1_vec = vfmaq_f32(C1_vec,A3_vec,B1_vec);
                C2_vec = vfmaq_f32(C2_vec,A3_vec,B2_vec);
                C3_vec = vfmaq_f32(C3_vec,A3_vec,B3_vec);

		B1_vec = vdupq_n_f32(B1[i+3]);
                B2_vec = vdupq_n_f32(B2[i+3]);
                B3_vec = vdupq_n_f32(B3[i+3]);

                C1_vec = vfmaq_f32(C1_vec,A4_vec,B1_vec);
                C2_vec = vfmaq_f32(C2_vec,A4_vec,B2_vec);
                C3_vec = vfmaq_f32(C3_vec,A4_vec,B3_vec);

		B1_vec = vdupq_n_f32(B1[i+4]);
                B2_vec = vdupq_n_f32(B2[i+4]);
                B3_vec = vdupq_n_f32(B3[i+4]);

                C1_vec = vfmaq_f32(C1_vec,A5_vec,B1_vec);
                C2_vec = vfmaq_f32(C2_vec,A5_vec,B2_vec);
                C3_vec = vfmaq_f32(C3_vec,A5_vec,B3_vec);

		
		vst1q_f32(&C1[i], C1_vec);
		vst1q_f32(&C2[i], C2_vec);
		vst1q_f32(&C3[i], C3_vec);

		}
	A_vec = (float32x4_t){ A[4], A[9], A[14], A[19] };
	for(int j = 0; j < n3; j++){
		i = 5 * j;
                B1_vec = vld1q_f32(&B1[i]);
                B2_vec = vld1q_f32(&B2[i]);
                B3_vec = vld1q_f32(&B3[i]);

		C1_vec = vmulq_f32(A_vec,B1_vec);
		C2_vec = vmulq_f32(A_vec,B2_vec);
		C3_vec = vmulq_f32(A_vec,B3_vec);
		
		C1[i+4] = vaddvq_f32(C1_vec);
		C1[i+4] += A[24]*B1[i+4];

		C2[i+4] = vaddvq_f32(C2_vec);
                C2[i+4] += A[24]*B2[i+4];

		C3[i+4] = vaddvq_f32(C3_vec);
                C3[i+4] += A[24]*B3[i+4];
	}
}

void mxm5_3comp_singleB_simd(float* A1,float* A2,float* A3,int n1,float* B,float* C1,float* C2,float* C3,int n3)
{
	int i = 0, k = 0;	
	float32x4_t A11_vec, A12_vec, A13_vec, A14_vec, A15_vec, B1_vec, B2_vec, B3_vec, B4_vec, B5_vec;
	float32x4_t A21_vec, A22_vec, A23_vec, A24_vec, A25_vec;
	float32x4_t A31_vec, A32_vec, A33_vec, A34_vec, A35_vec;
	float32x4_t A1_vec, A2_vec, A3_vec, B_vec;
        float32x4_t C1_vec, C2_vec, C3_vec;
	A1_vec = (float32x4_t){ A1[24], A1[49], A1[74], A1[99] };
        A2_vec = (float32x4_t){ A2[24], A2[49], A2[74], A2[99] };
        A3_vec = (float32x4_t){ A3[24], A3[49], A3[74], A3[99] };
	for(int j = 0; j < n3; j++){
		i = 5 * j;
		k = 25 * j;
		B1_vec = vdupq_n_f32(B[i]);
                B2_vec = vdupq_n_f32(B[i+1]);
                B3_vec = vdupq_n_f32(B[i+2]);
                B4_vec = vdupq_n_f32(B[i+3]);
                B5_vec = vdupq_n_f32(B[i+4]);
		for(int l = 0; l < 24; l+=4){
		A11_vec = vld1q_f32(&A1[l+0]);
		A12_vec = vld1q_f32(&A1[l+25]);
		A13_vec = vld1q_f32(&A1[l+50]);
		A14_vec = vld1q_f32(&A1[l+75]);
		A15_vec = vld1q_f32(&A1[l+100]);

		A21_vec = vld1q_f32(&A2[l+0]);
        	A22_vec = vld1q_f32(&A2[l+25]);
        	A23_vec = vld1q_f32(&A2[l+50]);
        	A24_vec = vld1q_f32(&A2[l+75]);
		A25_vec = vld1q_f32(&A2[l+100]);

		A31_vec = vld1q_f32(&A3[l+0]);
	        A32_vec = vld1q_f32(&A3[l+25]);
        	A33_vec = vld1q_f32(&A3[l+50]);
	        A34_vec = vld1q_f32(&A3[l+75]);
		A35_vec = vld1q_f32(&A3[l+100]);


		C1_vec = vmulq_f32(A11_vec,B1_vec);
		C2_vec = vmulq_f32(A21_vec,B1_vec);
		C3_vec = vmulq_f32(A31_vec,B1_vec);

		C1_vec = vfmaq_f32(C1_vec,A12_vec,B2_vec);
                C2_vec = vfmaq_f32(C2_vec,A22_vec,B2_vec);
                C3_vec = vfmaq_f32(C3_vec,A32_vec,B2_vec);

		C1_vec = vfmaq_f32(C1_vec,A13_vec,B3_vec);
                C2_vec = vfmaq_f32(C2_vec,A23_vec,B3_vec);
                C3_vec = vfmaq_f32(C3_vec,A33_vec,B3_vec);

		C1_vec = vfmaq_f32(C1_vec,A14_vec,B4_vec);
                C2_vec = vfmaq_f32(C2_vec,A24_vec,B4_vec);
                C3_vec = vfmaq_f32(C3_vec,A34_vec,B4_vec);

		C1_vec = vfmaq_f32(C1_vec,A15_vec,B5_vec);
                C2_vec = vfmaq_f32(C2_vec,A25_vec,B5_vec);
                C3_vec = vfmaq_f32(C3_vec,A35_vec,B5_vec);

		vst1q_f32(&C1[l+k], C1_vec);
                vst1q_f32(&C2[l+k], C2_vec);
                vst1q_f32(&C3[l+k], C3_vec);
	}
		B_vec = vld1q_f32(&B[i]);

		C1_vec = vmulq_f32(A1_vec,B_vec);
                C2_vec = vmulq_f32(A2_vec,B_vec);
                C3_vec = vmulq_f32(A3_vec,B_vec);

                C1[k+24] = vaddvq_f32(C1_vec);
                C1[k+24] += A1[124]*B[i+4];

                C2[k+24] = vaddvq_f32(C2_vec);
                C2[k+24] += A2[124]*B[i+4];

                C3[k+24] = vaddvq_f32(C3_vec);
                C3[k+24] += A3[124]*B[i+4];

		}
}

void mxm5_3comp_3dmat_single_simd(float* A1,float* A2,float* A3,int n1,float* B,int n2,float* C1,float* C2,float* C3,int n3)
{
	int i = 0, l = 0;
	float32x4_t A11_vec, A12_vec, A13_vec, A14_vec, A15_vec, B1_vec, B2_vec, B3_vec, B4_vec, B5_vec;
        float32x4_t A21_vec, A22_vec, A23_vec, A24_vec, A25_vec;
        float32x4_t A31_vec, A32_vec, A33_vec, A34_vec, A35_vec;
        float32x4_t A1_vec, A2_vec, A3_vec, B_vec;
        float32x4_t C1_vec, C2_vec, C3_vec;
	for(int k = 0; k < n3; k++){
		l = k * 25;
		A1_vec = (float32x4_t){ A1[k+4], A1[49], A1[74], A1[99] };
        	A2_vec = (float32x4_t){ A2[k+4], A2[49], A2[74], A2[99] };
	        A3_vec = (float32x4_t){ A3[k+4], A3[49], A3[74], A3[99] };

		A11_vec = vld1q_f32(&A1[l+0]);
                A12_vec = vld1q_f32(&A1[l+5]);
                A13_vec = vld1q_f32(&A1[l+10]);
                A14_vec = vld1q_f32(&A1[l+15]);
                A15_vec = vld1q_f32(&A1[l+20]);

                A21_vec = vld1q_f32(&A2[l+0]);
                A22_vec = vld1q_f32(&A2[l+5]);
                A23_vec = vld1q_f32(&A2[l+10]);
                A24_vec = vld1q_f32(&A2[l+15]);
                A25_vec = vld1q_f32(&A2[l+20]);

                A31_vec = vld1q_f32(&A3[l+0]);
                A32_vec = vld1q_f32(&A3[l+5]);
                A33_vec = vld1q_f32(&A3[l+10]);
                A34_vec = vld1q_f32(&A3[l+15]);
                A35_vec = vld1q_f32(&A3[l+20]);
		for(int j = 0; j < n2; j++){
			i = j * 5;
                        B1_vec = vdupq_n_f32(B[i]);
	                B2_vec = vdupq_n_f32(B[i+1]);
        	        B3_vec = vdupq_n_f32(B[i+2]);
                	B4_vec = vdupq_n_f32(B[i+3]);
	                B5_vec = vdupq_n_f32(B[i+4]);
			
			C1_vec = vmulq_f32(A11_vec,B1_vec);
                	C2_vec = vmulq_f32(A21_vec,B1_vec);
	                C3_vec = vmulq_f32(A31_vec,B1_vec);
	
        	        C1_vec = vfmaq_f32(C1_vec,A12_vec,B2_vec);
                	C2_vec = vfmaq_f32(C2_vec,A22_vec,B2_vec);
	                C3_vec = vfmaq_f32(C3_vec,A32_vec,B2_vec);
	
        	        C1_vec = vfmaq_f32(C1_vec,A13_vec,B3_vec);
                	C2_vec = vfmaq_f32(C2_vec,A23_vec,B3_vec);
	                C3_vec = vfmaq_f32(C3_vec,A33_vec,B3_vec);

        	        C1_vec = vfmaq_f32(C1_vec,A14_vec,B4_vec);
                	C2_vec = vfmaq_f32(C2_vec,A24_vec,B4_vec);
	                C3_vec = vfmaq_f32(C3_vec,A34_vec,B4_vec);

	                C1_vec = vfmaq_f32(C1_vec,A15_vec,B5_vec);
        	        C2_vec = vfmaq_f32(C2_vec,A25_vec,B5_vec);
                	C3_vec = vfmaq_f32(C3_vec,A35_vec,B5_vec);
			
			vst1q_f32(&C1[l+i], C1_vec);
	                vst1q_f32(&C2[l+i], C2_vec);
        	        vst1q_f32(&C3[l+i], C3_vec);

			B_vec = vld1q_f32(&B[i]);
			C1_vec = vmulq_f32(A1_vec,B_vec);
	                C2_vec = vmulq_f32(A2_vec,B_vec);
        	        C3_vec = vmulq_f32(A3_vec,B_vec);

                	C1[l+i+4] = vaddvq_f32(C1_vec);
	                C1[l+i+4] += A1[k+24]*B[i+4];

        	        C2[l+i+4] = vaddvq_f32(C2_vec);
                	C2[l+i+4] += A2[k+24]*B[i+4];

	                C3[l+i+4] = vaddvq_f32(C3_vec);
        	        C3[l+i+4] += A3[k+24]*B[i+4];
		}
	}
}
