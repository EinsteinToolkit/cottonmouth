#define CARPETX_GF3D5
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <loop_device.hxx>
#include <simd.hxx>
#include <defs.hxx>
#include <vect.hxx>
#include <cmath>
#include <tuple>
#include "../../../CarpetX/CarpetX/src/timer.hxx"
#ifdef __CUDACC__
#include <nvtx3/nvToolsExt.h>
#endif
#define access(GF, IDX) (GF(p.mask, IDX))
#define store(GF, IDX, VAL) (GF.store(p.mask, IDX, VAL))
#define stencil(GF, IDX) (GF(p.mask, IDX))
#define CCTK_ASSERT(X) if(!(X)) { CCTK_Error(__LINE__, __FILE__, CCTK_THORNSTRING, "Assertion Failure: " #X); }
using namespace Arith;
using namespace Loop;
using std::cbrt,std::fmax,std::fmin,std::sqrt;
void adm2bssn_pt2(CCTK_ARGUMENTS) {
    DECLARE_CCTK_ARGUMENTSX_adm2bssn_pt2;
    DECLARE_CCTK_PARAMETERS;
    using vreal = Arith::simd<CCTK_REAL>;
    constexpr std::size_t vsize = std::tuple_size_v<vreal>;
    #ifdef __CUDACC__
    const nvtxRangeId_t range = nvtxRangeStartA("adm2bssn_pt2");
    #endif
    const GF3D5layout VVV_layout(cctkGH, {0, 0, 0});
    #define ConfConnectU0_layout VVV_layout
    #define ConfConnectU1_layout VVV_layout
    #define ConfConnectU2_layout VVV_layout
    #define gtDD00_layout VVV_layout
    #define gtDD01_layout VVV_layout
    #define gtDD02_layout VVV_layout
    #define gtDD11_layout VVV_layout
    #define gtDD12_layout VVV_layout
    #define gtDD22_layout VVV_layout
    const auto DXI = (1.0 / CCTK_DELTA_SPACE(0));
    const auto DYI = (1.0 / CCTK_DELTA_SPACE(1));
    const auto DZI = (1.0 / CCTK_DELTA_SPACE(2));
    const vreal v_one = 1;
    const vreal v_zero = 0;
    grid.loop_int_device<VVV_centered[0], VVV_centered[1], VVV_centered[2], vsize>(grid.nghostzones, [=] CCTK_DEVICE(const PointDesc& p) CCTK_ATTRIBUTE_ALWAYS_INLINE {    
        const GF3D5index stencil_idx_m1_0_0_VVV(VVV_layout, p.I - p.DI[0]);
        const GF3D5index stencil_idx_m2_0_0_VVV(VVV_layout, p.I - 2*p.DI[0]);
        const GF3D5index stencil_idx_0_m1_0_VVV(VVV_layout, p.I - p.DI[1]);
        const GF3D5index stencil_idx_0_m2_0_VVV(VVV_layout, p.I - 2*p.DI[1]);
        const GF3D5index stencil_idx_0_0_m1_VVV(VVV_layout, p.I - p.DI[2]);
        const GF3D5index stencil_idx_0_0_m2_VVV(VVV_layout, p.I - 2*p.DI[2]);
        const GF3D5index stencil_idx_0_0_1_VVV(VVV_layout, p.I + p.DI[2]);
        const GF3D5index stencil_idx_0_0_2_VVV(VVV_layout, p.I + 2*p.DI[2]);
        const GF3D5index stencil_idx_0_1_0_VVV(VVV_layout, p.I + p.DI[1]);
        const GF3D5index stencil_idx_0_2_0_VVV(VVV_layout, p.I + 2*p.DI[1]);
        const GF3D5index stencil_idx_1_0_0_VVV(VVV_layout, p.I + p.DI[0]);
        const GF3D5index stencil_idx_2_0_0_VVV(VVV_layout, p.I + 2*p.DI[0]);
        const GF3D5index stencil_idx_0_0_0_VVV(VVV_layout, p.I);
        vreal x128 = stencil(gtDD12, stencil_idx_0_m2_0_VVV);
        vreal x129 = stencil(gtDD12, stencil_idx_0_2_0_VVV);
        vreal x130 = stencil(gtDD12, stencil_idx_0_m1_0_VVV);
        vreal x131 = stencil(gtDD12, stencil_idx_0_1_0_VVV);
        vreal x132 = (DYI * (((1.0 / 12.0) * ((x128 + (-(x129))))) + ((2.0 / 3.0) * ((x131 + (-(x130)))))));
        x128 = stencil(gtDD11, stencil_idx_0_0_m2_VVV);
        x129 = stencil(gtDD11, stencil_idx_0_0_2_VVV);
        x130 = stencil(gtDD11, stencil_idx_0_0_m1_VVV);
        x131 = stencil(gtDD11, stencil_idx_0_0_1_VVV);
        vreal x49 = (((1.0 / 12.0) * ((x128 + (-(x129))))) + ((2.0 / 3.0) * ((x131 + (-(x130))))));
        vreal x133 = (x132 + ((-1.0 / 2.0) * DZI * x49));
        vreal x137 = (access(gtDD11, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x138 = pow2(access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x189 = (((1.0 / 2.0) * x137) + ((-1.0 / 2.0) * x138));
        vreal x73 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x74 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x200 = (x73 + (-(x74)));
        vreal x110 = stencil(gtDD11, stencil_idx_0_m2_0_VVV);
        vreal x111 = stencil(gtDD11, stencil_idx_0_2_0_VVV);
        vreal x112 = stencil(gtDD11, stencil_idx_0_m1_0_VVV);
        vreal x113 = stencil(gtDD11, stencil_idx_0_1_0_VVV);
        vreal x114 = (DYI * (((1.0 / 12.0) * ((x110 + (-(x111))))) + ((2.0 / 3.0) * ((x113 + (-(x112)))))));
        x110 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        x111 = (((1.0 / 2.0) * x110) + ((-1.0 / 2.0) * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)));
        x112 = (-(x111));
        x113 = (x112 * x114);
        vreal x100 = stencil(gtDD11, stencil_idx_m2_0_0_VVV);
        vreal x101 = stencil(gtDD11, stencil_idx_2_0_0_VVV);
        vreal x102 = stencil(gtDD11, stencil_idx_m1_0_0_VVV);
        vreal x103 = stencil(gtDD11, stencil_idx_1_0_0_VVV);
        vreal x104 = (DXI * (((1.0 / 12.0) * ((x100 + (-(x101))))) + ((2.0 / 3.0) * ((x103 + (-(x102)))))));
        x100 = stencil(gtDD01, stencil_idx_0_m2_0_VVV);
        x101 = stencil(gtDD01, stencil_idx_0_2_0_VVV);
        x102 = stencil(gtDD01, stencil_idx_0_m1_0_VVV);
        x103 = stencil(gtDD01, stencil_idx_0_1_0_VVV);
        vreal x122 = (((1.0 / 12.0) * ((x100 + (-(x101))))) + ((2.0 / 3.0) * ((x103 + (-(x102))))));
        vreal x207 = (x104 + (-2 * DYI * x122));
        vreal x208 = (-(x207));
        x207 = (x113 + (x133 * x200) + (x189 * x208));
        x208 = stencil(gtDD22, stencil_idx_0_m2_0_VVV);
        vreal x37 = stencil(gtDD22, stencil_idx_0_2_0_VVV);
        vreal x38 = stencil(gtDD22, stencil_idx_0_m1_0_VVV);
        vreal x39 = stencil(gtDD22, stencil_idx_0_1_0_VVV);
        vreal x40 = (DYI * (((1.0 / 12.0) * ((x208 + (-(x37))))) + ((2.0 / 3.0) * ((x39 + (-(x38)))))));
        x37 = stencil(gtDD12, stencil_idx_0_0_m2_VVV);
        x38 = stencil(gtDD12, stencil_idx_0_0_2_VVV);
        x39 = stencil(gtDD12, stencil_idx_0_0_m1_VVV);
        vreal x181 = stencil(gtDD12, stencil_idx_0_0_1_VVV);
        vreal x182 = (((1.0 / 12.0) * ((x37 + (-(x38))))) + ((2.0 / 3.0) * ((x181 + (-(x39))))));
        x181 = (DZI * x182);
        vreal x212 = (((1.0 / 2.0) * x40) + (-(x181)));
        vreal x116 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x117 = (x110 + (-(x116)));
        x116 = stencil(gtDD22, stencil_idx_0_0_m2_VVV);
        vreal x166 = stencil(gtDD22, stencil_idx_0_0_2_VVV);
        vreal x167 = stencil(gtDD22, stencil_idx_0_0_m1_VVV);
        vreal x168 = stencil(gtDD22, stencil_idx_0_0_1_VVV);
        vreal x169 = (DZI * (((1.0 / 12.0) * ((x116 + (-(x166))))) + ((2.0 / 3.0) * ((x168 + (-(x167)))))));
        x166 = (((1.0 / 2.0) * x73) + ((-1.0 / 2.0) * x74));
        x167 = (x166 * x169);
        x168 = stencil(gtDD22, stencil_idx_m2_0_0_VVV);
        vreal x77 = stencil(gtDD22, stencil_idx_2_0_0_VVV);
        vreal x78 = stencil(gtDD22, stencil_idx_m1_0_0_VVV);
        vreal x79 = stencil(gtDD22, stencil_idx_1_0_0_VVV);
        vreal x80 = (DXI * (((1.0 / 12.0) * ((x168 + (-(x77))))) + ((2.0 / 3.0) * ((x79 + (-(x78)))))));
        x77 = stencil(gtDD02, stencil_idx_0_0_m2_VVV);
        x78 = stencil(gtDD02, stencil_idx_0_0_2_VVV);
        x79 = stencil(gtDD02, stencil_idx_0_0_m1_VVV);
        vreal x174 = stencil(gtDD02, stencil_idx_0_0_1_VVV);
        vreal x175 = (((1.0 / 12.0) * ((x77 + (-(x78))))) + ((2.0 / 3.0) * ((x174 + (-(x79))))));
        x174 = (x80 + (-2 * DZI * x175));
        vreal x214 = (-(x174));
        vreal x215 = (x167 + (x117 * x212) + (x189 * x214));
        x214 = stencil(gtDD02, stencil_idx_0_m2_0_VVV);
        vreal x56 = stencil(gtDD02, stencil_idx_0_2_0_VVV);
        vreal x57 = stencil(gtDD02, stencil_idx_0_m1_0_VVV);
        vreal x58 = stencil(gtDD02, stencil_idx_0_1_0_VVV);
        vreal x59 = (DYI * (((1.0 / 12.0) * ((x214 + (-(x56))))) + ((2.0 / 3.0) * ((x58 + (-(x57)))))));
        x56 = stencil(gtDD01, stencil_idx_0_0_m2_VVV);
        x57 = stencil(gtDD01, stencil_idx_0_0_2_VVV);
        x58 = stencil(gtDD01, stencil_idx_0_0_m1_VVV);
        vreal x63 = stencil(gtDD01, stencil_idx_0_0_1_VVV);
        vreal x64 = (DZI * (((1.0 / 12.0) * ((x56 + (-(x57))))) + ((2.0 / 3.0) * ((x63 + (-(x58)))))));
        x63 = stencil(gtDD12, stencil_idx_m2_0_0_VVV);
        vreal x66 = stencil(gtDD12, stencil_idx_2_0_0_VVV);
        vreal x67 = stencil(gtDD12, stencil_idx_m1_0_0_VVV);
        vreal x68 = stencil(gtDD12, stencil_idx_1_0_0_VVV);
        vreal x69 = (DXI * (((1.0 / 12.0) * ((x63 + (-(x66))))) + ((2.0 / 3.0) * ((x68 + (-(x67)))))));
        x66 = (x59 + x64 + (-(x69)));
        x67 = (x166 * x40);
        x68 = (DZI * x49);
        vreal x188 = (x112 * x68);
        vreal x190 = (x188 + x67 + (x189 * x66));
        x188 = (x166 * x80);
        vreal x82 = stencil(gtDD00, stencil_idx_0_0_m2_VVV);
        vreal x83 = stencil(gtDD00, stencil_idx_0_0_2_VVV);
        vreal x84 = stencil(gtDD00, stencil_idx_0_0_m1_VVV);
        vreal x85 = stencil(gtDD00, stencil_idx_0_0_1_VVV);
        vreal x86 = (((1.0 / 12.0) * ((x82 + (-(x83))))) + ((2.0 / 3.0) * ((x85 + (-(x84))))));
        x82 = (DZI * x86);
        x83 = (x189 * x82);
        x84 = (x64 + x69 + (-(x59)));
        x85 = (x112 * x84);
        vreal x194 = (x188 + x83 + x85);
        vreal x195 = (x104 * x112);
        vreal x93 = stencil(gtDD00, stencil_idx_0_m2_0_VVV);
        vreal x94 = stencil(gtDD00, stencil_idx_0_2_0_VVV);
        vreal x95 = stencil(gtDD00, stencil_idx_0_m1_0_VVV);
        vreal x96 = stencil(gtDD00, stencil_idx_0_1_0_VVV);
        vreal x97 = (((1.0 / 12.0) * ((x93 + (-(x94))))) + ((2.0 / 3.0) * ((x96 + (-(x95))))));
        x93 = (DYI * x97);
        x94 = (x189 * x93);
        x95 = (x59 + x69 + (-(x64)));
        x59 = (x166 * x95);
        x64 = (x195 + x59 + x94);
        x195 = stencil(gtDD00, stencil_idx_m2_0_0_VVV);
        x69 = stencil(gtDD00, stencil_idx_2_0_0_VVV);
        x96 = stencil(gtDD00, stencil_idx_m1_0_0_VVV);
        vreal x143 = stencil(gtDD00, stencil_idx_1_0_0_VVV);
        vreal x144 = (DXI * (((1.0 / 12.0) * ((x195 + (-(x69))))) + ((2.0 / 3.0) * ((x143 + (-(x96)))))));
        x143 = (x144 * x189);
        x189 = stencil(gtDD02, stencil_idx_m2_0_0_VVV);
        vreal x147 = stencil(gtDD02, stencil_idx_2_0_0_VVV);
        vreal x148 = stencil(gtDD02, stencil_idx_m1_0_0_VVV);
        vreal x149 = stencil(gtDD02, stencil_idx_1_0_0_VVV);
        vreal x150 = (DXI * (((1.0 / 12.0) * ((x189 + (-(x147))))) + ((2.0 / 3.0) * ((x149 + (-(x148)))))));
        x147 = (x150 + ((-1.0 / 2.0) * DZI * x86));
        x148 = (x147 * x200);
        x149 = stencil(gtDD01, stencil_idx_m2_0_0_VVV);
        vreal x155 = stencil(gtDD01, stencil_idx_2_0_0_VVV);
        vreal x156 = stencil(gtDD01, stencil_idx_m1_0_0_VVV);
        vreal x157 = stencil(gtDD01, stencil_idx_1_0_0_VVV);
        vreal x158 = (DXI * (((1.0 / 12.0) * ((x149 + (-(x155))))) + ((2.0 / 3.0) * ((x157 + (-(x156)))))));
        x155 = (x158 + ((-1.0 / 2.0) * DYI * x97));
        x156 = (-(x155));
        x157 = (x117 * x156);
        vreal x205 = (x143 + x148 + x157);
        vreal x75 = ((-2 * x74) + (2 * x73));
        x73 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        x74 = pow2(access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x109 = (x73 + (-(x74)));
        vreal x139 = (x137 + (-(x138)));
        x137 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        x138 = pow2(access(gtDD01, stencil_idx_0_0_0_VVV));
        vreal x164 = (x137 + (-(x138)));
        vreal x31 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x32 = ((2 * x31) + (-2 * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV)));
        vreal x33 = (-(x32));
        x32 = ((2 * x110) + (-2 * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)));
        vreal x92 = (-(x32));
        store(ConfConnectU0, stencil_idx_0_0_0_VVV, ((x109 * x207) + (x139 * x205) + (x164 * x215) + (x190 * x33) + (x194 * x75) + (x64 * x92)));
        x190 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV));
        x194 = (x31 + (-(x190)));
        x205 = (x112 * x144);
        x215 = (-(x147));
        vreal x44 = (((1.0 / 2.0) * x73) + ((-1.0 / 2.0) * x74));
        vreal x153 = (-(x44));
        vreal x159 = ((2 * x158) + (-1 * DYI * x97));
        x158 = (-(x159));
        x159 = (x205 + (x153 * x158) + (x194 * x215));
        x153 = (DZI * x175);
        x175 = (((1.0 / 2.0) * x80) + (-(x153)));
        x97 = (((1.0 / 2.0) * x31) + ((-1.0 / 2.0) * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV)));
        x31 = (-(x97));
        vreal x170 = (x169 * x31);
        vreal x183 = (x40 + (-2 * DZI * x182));
        x182 = (-(x183));
        x183 = (x170 + (x117 * x175) + (x182 * x44));
        x170 = (x31 * x80);
        vreal x88 = (x112 * x82);
        vreal x90 = (x170 + x88 + (x44 * x84));
        x88 = (x104 * x44);
        vreal x107 = (x31 * x95);
        vreal x99 = (x112 * x93);
        vreal x108 = (x107 + x88 + x99);
        x107 = (x114 * x44);
        x99 = (DYI * x122);
        x122 = (((1.0 / 2.0) * x104) + (-(x99)));
        vreal x125 = (x117 * x122);
        x117 = (-(x133));
        x133 = (x117 * x194);
        vreal x136 = (x107 + x125 + x133);
        x125 = (x31 * x40);
        vreal x51 = (x44 * x68);
        x44 = (x112 * x66);
        vreal x72 = (x125 + x44 + x51);
        store(ConfConnectU1, stencil_idx_0_0_0_VVV, ((x108 * x92) + (x109 * x136) + (x139 * x159) + (x164 * x183) + (x33 * x72) + (x75 * x90)));
        x108 = ((-2 * x132) + (DZI * x49));
        x132 = (x114 * x31);
        x114 = (-(x122));
        x49 = (((1.0 / 2.0) * x137) + ((-1.0 / 2.0) * x138));
        x136 = (-(x49));
        x72 = (x132 + (x108 * x136) + (x114 * x200));
        x90 = (x144 * x166);
        x144 = ((2 * x150) + (-1 * DZI * x86));
        x150 = (-(x144));
        x86 = (x90 + (x136 * x150) + (x156 * x194));
        x51 = (x104 * x31);
        x104 = (x166 * x93);
        vreal x227 = (x104 + x51 + (x49 * x95));
        vreal x216 = (x31 * x68);
        vreal x218 = (x40 * x49);
        x40 = (x166 * x66);
        vreal x220 = (x216 + x218 + x40);
        x216 = (x166 * x82);
        x218 = (x49 * x80);
        x80 = (x31 * x84);
        vreal x224 = (x216 + x218 + x80);
        vreal x228 = (x169 * x49);
        x169 = (-(x175));
        vreal x230 = (x169 * x200);
        x200 = (x194 * x212);
        x212 = (x200 + x228 + x230);
        store(ConfConnectU2, stencil_idx_0_0_0_VVV, ((x109 * x72) + (x139 * x86) + (x164 * x212) + (x220 * x33) + (x224 * x75) + (x227 * x92)));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}