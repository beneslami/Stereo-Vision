#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "x86intrin.h"

using namespace cv;
using namespace std;

#define W 3
#define B 45
#define _mm_cmple_epu16(a, b) _mm_cmpeq_epi16(_mm_min_epu16(a,b),a)
#define _mm_min_pos_epu16(Pre, mask, i) _mm_min_epu16(_mm_set1_epi16(i), _mm_or_si128(mask,Pre))

int main(){
    unsigned char *pRes;
    unsigned char *pSrcR;
    unsigned char *pSrcL;
    pRes = &out_img2.at<uchar>(0,0);
    pSrcR = &in_imgR.at<uchar>(0,0);
    pSrcL = &in_imgL.at<uchar>(0,0);

    __m128i zero = _mm_setzero_si128();
    __m128i tst;
    __m128i m11r, m12r, m13r, m21r, m22r, m23r, m31r, m32r, m33r;
    __m128i m11l, m12l, m13l, m21l, m22l, m23l, m31l, m32l, m33l;
    __m128i m11o, m12o, m13o, m21o, m22o, m23o, m31o, m32o, m33o;
    __m128i m11oh, m12oh, m13oh, m21oh, m22oh, m23oh, m31oh, m32oh, m33oh;

    __m128i Result,Res;
    __m128i Resulth,maskh,mask;
    __m128i minih,mini,min_posh,min_pos;

    start = clock();
    for (int k = 0; k < height; k++)
    {
        for (int l = 0; l < width; l+=16)
        {
            m11r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l));
            m12r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + 1));
            m13r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + 2));
            m21r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + width));
            m22r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + width+1));
            m23r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + width+2));
            m31r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + 2*width));
            m32r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + 2*width+1));
            m33r = _mm_loadu_si128((const __m128i*)(pSrcR + (k*width) + l + 2*width+2));

            m11l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l));
            m12l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 1));
            m13l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 2));
            m21l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + width));
            m22l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + width+1));
            m23l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + width+2));
            m31l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 2*width));
            m32l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 2*width+1));
            m33l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 2*width+2));

            m11o = _mm_abs_epi8(_mm_sub_epi8(m11l,m11r));
            m12o = _mm_abs_epi8(_mm_sub_epi8(m12l,m12r));
            m13o = _mm_abs_epi8(_mm_sub_epi8(m13l,m13r));
            m21o = _mm_abs_epi8(_mm_sub_epi8(m21l,m21r));
            m22o = _mm_abs_epi8(_mm_sub_epi8(m22l,m22r));
            m23o = _mm_abs_epi8(_mm_sub_epi8(m23l,m23r));
            m31o = _mm_abs_epi8(_mm_sub_epi8(m31l,m31r));
            m32o = _mm_abs_epi8(_mm_sub_epi8(m32l,m32r));
            m33o = _mm_abs_epi8(_mm_sub_epi8(m33l,m33r));

            m11oh = _mm_unpackhi_epi8(m11o, zero);
            m12oh = _mm_unpackhi_epi8(m12o, zero);
            m13oh = _mm_unpackhi_epi8(m13o, zero);
            m21oh = _mm_unpackhi_epi8(m21o, zero);
            m22oh = _mm_unpackhi_epi8(m22o, zero);
            m23oh = _mm_unpackhi_epi8(m23o, zero);
            m31oh = _mm_unpackhi_epi8(m31o, zero);
            m32oh = _mm_unpackhi_epi8(m32o, zero);
            m33oh = _mm_unpackhi_epi8(m33o, zero);
            m11o = _mm_unpacklo_epi8(m11o, zero);
            m12o = _mm_unpacklo_epi8(m12o, zero);
            m13o = _mm_unpacklo_epi8(m13o, zero);
            m21o = _mm_unpacklo_epi8(m21o, zero);
            m22o = _mm_unpacklo_epi8(m22o, zero);
            m23o = _mm_unpacklo_epi8(m23o, zero);
            m31o = _mm_unpacklo_epi8(m31o, zero);
            m32o = _mm_unpacklo_epi8(m32o, zero);
            m33o = _mm_unpacklo_epi8(m33o, zero);

            Result = m11o;
            Result = _mm_add_epi16(Result,m12o);
            Result = _mm_add_epi16(Result,m13o);
            Result = _mm_add_epi16(Result,m21o);
            Result = _mm_add_epi16(Result,m22o);
            Result = _mm_add_epi16(Result,m23o);
            Result = _mm_add_epi16(Result,m31o);
            Result = _mm_add_epi16(Result,m32o);
            Result = _mm_add_epi16(Result,m33o);

            Resulth = m11oh;
            Resulth = _mm_add_epi16(Resulth,m12oh);
            Resulth = _mm_add_epi16(Resulth,m13oh);
            Resulth = _mm_add_epi16(Resulth,m21oh);
            Resulth = _mm_add_epi16(Resulth,m22oh);
            Resulth = _mm_add_epi16(Resulth,m23oh);
            Resulth = _mm_add_epi16(Resulth,m31oh);
            Resulth = _mm_add_epi16(Resulth,m32oh);
            Resulth = _mm_add_epi16(Resulth,m33oh);

            mini = Result;
            min_pos = zero;

            minih = Resulth;
            min_posh = zero;

            for (int i = 1; i < B; i++)
            {
                m11l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + i));
                m12l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + i+1));
                m13l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + i+2));
                m21l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + i+width));
                m22l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + i+width+1));
                m23l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + i+width+2));
                m31l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 2*width+i));
                m32l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 2*width+i+1));
                m33l = _mm_loadu_si128((const __m128i*)(pSrcL + (k*width) + l + 2*width+i+2));

                m11o = _mm_abs_epi8(_mm_sub_epi8(m11l,m11r));
                m12o = _mm_abs_epi8(_mm_sub_epi8(m12l,m12r));
                m13o = _mm_abs_epi8(_mm_sub_epi8(m13l,m13r));
                m21o = _mm_abs_epi8(_mm_sub_epi8(m21l,m21r));
                m22o = _mm_abs_epi8(_mm_sub_epi8(m22l,m22r));
                m23o = _mm_abs_epi8(_mm_sub_epi8(m23l,m23r));
                m31o = _mm_abs_epi8(_mm_sub_epi8(m31l,m31r));
                m32o = _mm_abs_epi8(_mm_sub_epi8(m32l,m32r));
                m33o = _mm_abs_epi8(_mm_sub_epi8(m33l,m33r));

                m11oh = _mm_unpackhi_epi8(m11o, zero);
                m12oh = _mm_unpackhi_epi8(m12o, zero);
                m13oh = _mm_unpackhi_epi8(m13o, zero);
                m21oh = _mm_unpackhi_epi8(m21o, zero);
                m22oh = _mm_unpackhi_epi8(m22o, zero);
                m23oh = _mm_unpackhi_epi8(m23o, zero);
                m31oh = _mm_unpackhi_epi8(m31o, zero);
                m32oh = _mm_unpackhi_epi8(m32o, zero);
                m33oh = _mm_unpackhi_epi8(m33o, zero);
                m11o = _mm_unpacklo_epi8(m11o, zero);
                m12o = _mm_unpacklo_epi8(m12o, zero);
                m13o = _mm_unpacklo_epi8(m13o, zero);
                m21o = _mm_unpacklo_epi8(m21o, zero);
                m22o = _mm_unpacklo_epi8(m22o, zero);
                m23o = _mm_unpacklo_epi8(m23o, zero);
                m31o = _mm_unpacklo_epi8(m31o, zero);
                m32o = _mm_unpacklo_epi8(m32o, zero);
                m33o = _mm_unpacklo_epi8(m33o, zero);

                Result = m11o;
                Result = _mm_add_epi16(Result,m12o);
                Result = _mm_add_epi16(Result,m13o);
                Result = _mm_add_epi16(Result,m21o);
                Result = _mm_add_epi16(Result,m22o);
                Result = _mm_add_epi16(Result,m23o);
                Result = _mm_add_epi16(Result,m31o);
                Result = _mm_add_epi16(Result,m32o);
                Result = _mm_add_epi16(Result,m33o);

                Resulth = m11oh;
                Resulth = _mm_add_epi16(Resulth,m12oh);
                Resulth = _mm_add_epi16(Resulth,m13oh);
                Resulth = _mm_add_epi16(Resulth,m21oh);
                Resulth = _mm_add_epi16(Resulth,m22oh);
                Resulth = _mm_add_epi16(Resulth,m23oh);
                Resulth = _mm_add_epi16(Resulth,m31oh);
                Resulth = _mm_add_epi16(Resulth,m32oh);
                Resulth = _mm_add_epi16(Resulth,m33oh);

                mask = _mm_cmple_epu16(Result, mini);
                mini = _mm_min_epu16(Result, mini);
                min_pos = _mm_min_pos_epu16(min_pos, mask, i);

                maskh = _mm_cmple_epu16(Resulth, minih);
                minih = _mm_min_epu16(Resulth, minih);
                min_posh = _mm_min_pos_epu16(min_posh, maskh, i);
            }
            tst = _mm_packs_epi16(min_pos,min_posh);
            tst = _mm_slli_epi16(tst,2);
            _mm_storeu_si128((__m128i*)(pRes + (k*width) + l + width + 1), tst);
        }
    }


    end   = clock();
    cout << time2 = end - start << endl;
    return 0;
}
