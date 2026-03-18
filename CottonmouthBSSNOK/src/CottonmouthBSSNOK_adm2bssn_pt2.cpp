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
        vreal x36 = stencil(gtDD22, stencil_idx_0_m2_0_VVV);
        vreal x37 = stencil(gtDD22, stencil_idx_0_2_0_VVV);
        vreal x38 = stencil(gtDD22, stencil_idx_0_m1_0_VVV);
        vreal x39 = stencil(gtDD22, stencil_idx_0_1_0_VVV);
        vreal x45 = stencil(gtDD11, stencil_idx_0_0_m2_VVV);
        vreal x46 = stencil(gtDD11, stencil_idx_0_0_2_VVV);
        vreal x47 = stencil(gtDD11, stencil_idx_0_0_m1_VVV);
        vreal x48 = stencil(gtDD11, stencil_idx_0_0_1_VVV);
        vreal x55 = stencil(gtDD02, stencil_idx_0_m2_0_VVV);
        vreal x56 = stencil(gtDD02, stencil_idx_0_2_0_VVV);
        vreal x57 = stencil(gtDD02, stencil_idx_0_m1_0_VVV);
        vreal x58 = stencil(gtDD02, stencil_idx_0_1_0_VVV);
        vreal x60 = stencil(gtDD01, stencil_idx_0_0_m2_VVV);
        vreal x61 = stencil(gtDD01, stencil_idx_0_0_2_VVV);
        vreal x62 = stencil(gtDD01, stencil_idx_0_0_m1_VVV);
        vreal x63 = stencil(gtDD01, stencil_idx_0_0_1_VVV);
        vreal x82 = stencil(gtDD00, stencil_idx_0_0_m2_VVV);
        vreal x83 = stencil(gtDD00, stencil_idx_0_0_2_VVV);
        vreal x84 = stencil(gtDD00, stencil_idx_0_0_m1_VVV);
        vreal x85 = stencil(gtDD00, stencil_idx_0_0_1_VVV);
        vreal x93 = stencil(gtDD00, stencil_idx_0_m2_0_VVV);
        vreal x94 = stencil(gtDD00, stencil_idx_0_2_0_VVV);
        vreal x95 = stencil(gtDD00, stencil_idx_0_m1_0_VVV);
        vreal x96 = stencil(gtDD00, stencil_idx_0_1_0_VVV);
        vreal x110 = stencil(gtDD11, stencil_idx_0_m2_0_VVV);
        vreal x111 = stencil(gtDD11, stencil_idx_0_2_0_VVV);
        vreal x112 = stencil(gtDD11, stencil_idx_0_m1_0_VVV);
        vreal x113 = stencil(gtDD11, stencil_idx_0_1_0_VVV);
        vreal x118 = stencil(gtDD01, stencil_idx_0_m2_0_VVV);
        vreal x119 = stencil(gtDD01, stencil_idx_0_2_0_VVV);
        vreal x120 = stencil(gtDD01, stencil_idx_0_m1_0_VVV);
        vreal x121 = stencil(gtDD01, stencil_idx_0_1_0_VVV);
        vreal x128 = stencil(gtDD12, stencil_idx_0_m2_0_VVV);
        vreal x129 = stencil(gtDD12, stencil_idx_0_2_0_VVV);
        vreal x130 = stencil(gtDD12, stencil_idx_0_m1_0_VVV);
        vreal x131 = stencil(gtDD12, stencil_idx_0_1_0_VVV);
        vreal x165 = stencil(gtDD22, stencil_idx_0_0_m2_VVV);
        vreal x166 = stencil(gtDD22, stencil_idx_0_0_2_VVV);
        vreal x167 = stencil(gtDD22, stencil_idx_0_0_m1_VVV);
        vreal x168 = stencil(gtDD22, stencil_idx_0_0_1_VVV);
        vreal x171 = stencil(gtDD02, stencil_idx_0_0_m2_VVV);
        vreal x172 = stencil(gtDD02, stencil_idx_0_0_2_VVV);
        vreal x173 = stencil(gtDD02, stencil_idx_0_0_m1_VVV);
        vreal x174 = stencil(gtDD02, stencil_idx_0_0_1_VVV);
        vreal x178 = stencil(gtDD12, stencil_idx_0_0_m2_VVV);
        vreal x179 = stencil(gtDD12, stencil_idx_0_0_2_VVV);
        vreal x180 = stencil(gtDD12, stencil_idx_0_0_m1_VVV);
        vreal x181 = stencil(gtDD12, stencil_idx_0_0_1_VVV);
        vreal x65 = stencil(gtDD12, stencil_idx_m2_0_0_VVV);
        vreal x66 = stencil(gtDD12, stencil_idx_2_0_0_VVV);
        vreal x67 = stencil(gtDD12, stencil_idx_m1_0_0_VVV);
        vreal x68 = stencil(gtDD12, stencil_idx_1_0_0_VVV);
        vreal x76 = stencil(gtDD22, stencil_idx_m2_0_0_VVV);
        vreal x77 = stencil(gtDD22, stencil_idx_2_0_0_VVV);
        vreal x78 = stencil(gtDD22, stencil_idx_m1_0_0_VVV);
        vreal x79 = stencil(gtDD22, stencil_idx_1_0_0_VVV);
        vreal x100 = stencil(gtDD11, stencil_idx_m2_0_0_VVV);
        vreal x101 = stencil(gtDD11, stencil_idx_2_0_0_VVV);
        vreal x102 = stencil(gtDD11, stencil_idx_m1_0_0_VVV);
        vreal x103 = stencil(gtDD11, stencil_idx_1_0_0_VVV);
        vreal x140 = stencil(gtDD00, stencil_idx_m2_0_0_VVV);
        vreal x141 = stencil(gtDD00, stencil_idx_2_0_0_VVV);
        vreal x142 = stencil(gtDD00, stencil_idx_m1_0_0_VVV);
        vreal x143 = stencil(gtDD00, stencil_idx_1_0_0_VVV);
        vreal x146 = stencil(gtDD02, stencil_idx_m2_0_0_VVV);
        vreal x147 = stencil(gtDD02, stencil_idx_2_0_0_VVV);
        vreal x148 = stencil(gtDD02, stencil_idx_m1_0_0_VVV);
        vreal x149 = stencil(gtDD02, stencil_idx_1_0_0_VVV);
        vreal x154 = stencil(gtDD01, stencil_idx_m2_0_0_VVV);
        vreal x155 = stencil(gtDD01, stencil_idx_2_0_0_VVV);
        vreal x156 = stencil(gtDD01, stencil_idx_m1_0_0_VVV);
        vreal x157 = stencil(gtDD01, stencil_idx_1_0_0_VVV);
        vreal x31 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x32 = ((2 * x31) + (-2 * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV)));
        vreal x34 = (((1.0 / 2.0) * x31) + ((-1.0 / 2.0) * access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV)));
        vreal x52 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x53 = (((1.0 / 2.0) * x52) + ((-1.0 / 2.0) * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)));
        vreal x91 = ((2 * x52) + (-2 * access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV)));
        vreal x42 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x73 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x74 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x116 = (access(gtDD02, stencil_idx_0_0_0_VVV) * access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x126 = (access(gtDD01, stencil_idx_0_0_0_VVV) * access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x137 = (access(gtDD11, stencil_idx_0_0_0_VVV) * access(gtDD22, stencil_idx_0_0_0_VVV));
        vreal x162 = (access(gtDD00, stencil_idx_0_0_0_VVV) * access(gtDD11, stencil_idx_0_0_0_VVV));
        vreal x43 = pow2(access(gtDD02, stencil_idx_0_0_0_VVV));
        vreal x138 = pow2(access(gtDD12, stencil_idx_0_0_0_VVV));
        vreal x163 = pow2(access(gtDD01, stencil_idx_0_0_0_VVV));
        vreal x54 = (-(x53));
        x53 = (DXI * (((1.0 / 12.0) * ((x65 + (-(x66))))) + ((2.0 / 3.0) * ((x68 + (-(x67)))))));
        x65 = (DYI * (((1.0 / 12.0) * ((x55 + (-(x56))))) + ((2.0 / 3.0) * ((x58 + (-(x57)))))));
        x55 = (DZI * (((1.0 / 12.0) * ((x60 + (-(x61))))) + ((2.0 / 3.0) * ((x63 + (-(x62)))))));
        x60 = (x53 + x55 + (-(x65)));
        x61 = (x54 * x60);
        x62 = (((1.0 / 12.0) * ((x82 + (-(x83))))) + ((2.0 / 3.0) * ((x85 + (-(x84))))));
        x82 = (DZI * x62);
        x83 = (((1.0 / 2.0) * x137) + ((-1.0 / 2.0) * x138));
        x84 = (x82 * x83);
        x85 = (((1.0 / 2.0) * x73) + ((-1.0 / 2.0) * x74));
        x63 = (DXI * (((1.0 / 12.0) * ((x76 + (-(x77))))) + ((2.0 / 3.0) * ((x79 + (-(x78)))))));
        x76 = (x63 * x85);
        x77 = (x61 + x76 + x84);
        x78 = (x137 + (-(x138)));
        x137 = (x42 + (-(x43)));
        x138 = (DYI * (((1.0 / 12.0) * ((x110 + (-(x111))))) + ((2.0 / 3.0) * ((x113 + (-(x112)))))));
        x110 = (x138 * x54);
        x111 = (DYI * (((1.0 / 12.0) * ((x128 + (-(x129))))) + ((2.0 / 3.0) * ((x131 + (-(x130)))))));
        x128 = (((1.0 / 12.0) * ((x45 + (-(x46))))) + ((2.0 / 3.0) * ((x48 + (-(x47))))));
        x45 = (x111 + ((-1.0 / 2.0) * DZI * x128));
        x46 = (x73 + (-(x74)));
        x47 = (DXI * (((1.0 / 12.0) * ((x100 + (-(x101))))) + ((2.0 / 3.0) * ((x103 + (-(x102)))))));
        x100 = (((1.0 / 12.0) * ((x118 + (-(x119))))) + ((2.0 / 3.0) * ((x121 + (-(x120))))));
        x118 = (x47 + (-2 * DYI * x100));
        x119 = (-(x118));
        x120 = (x110 + (x119 * x83) + (x45 * x46));
        x121 = (DXI * (((1.0 / 12.0) * ((x146 + (-(x147))))) + ((2.0 / 3.0) * ((x149 + (-(x148)))))));
        x146 = (x121 + ((-1.0 / 2.0) * DZI * x62));
        x147 = (x146 * x46);
        x148 = (DXI * (((1.0 / 12.0) * ((x140 + (-(x141))))) + ((2.0 / 3.0) * ((x143 + (-(x142)))))));
        x140 = (x148 * x83);
        x141 = (DXI * (((1.0 / 12.0) * ((x154 + (-(x155))))) + ((2.0 / 3.0) * ((x157 + (-(x156)))))));
        x154 = (((1.0 / 12.0) * ((x93 + (-(x94))))) + ((2.0 / 3.0) * ((x96 + (-(x95))))));
        x93 = (x141 + ((-1.0 / 2.0) * DYI * x154));
        x94 = (-(x93));
        x95 = (x52 + (-(x116)));
        x116 = (x94 * x95);
        x52 = (x116 + x140 + x147);
        x96 = (((1.0 / 12.0) * ((x171 + (-(x172))))) + ((2.0 / 3.0) * ((x174 + (-(x173))))));
        x171 = (x63 + (-2 * DZI * x96));
        x172 = (-(x171));
        x173 = (DYI * (((1.0 / 12.0) * ((x36 + (-(x37))))) + ((2.0 / 3.0) * ((x39 + (-(x38)))))));
        x36 = (((1.0 / 12.0) * ((x178 + (-(x179))))) + ((2.0 / 3.0) * ((x181 + (-(x180))))));
        x178 = (DZI * x36);
        x179 = (((1.0 / 2.0) * x173) + (-(x178)));
        x180 = (DZI * (((1.0 / 12.0) * ((x165 + (-(x166))))) + ((2.0 / 3.0) * ((x168 + (-(x167)))))));
        x165 = (x180 * x85);
        x166 = (x165 + (x172 * x83) + (x179 * x95));
        x167 = (x173 * x85);
        x168 = (DZI * x128);
        x181 = (x168 * x54);
        x37 = (x55 + x65 + (-(x53)));
        x38 = (x167 + x181 + (x37 * x83));
        x39 = (-(x32));
        x32 = ((-2 * x74) + (2 * x73));
        x73 = (x47 * x54);
        x74 = (x53 + x65 + (-(x55)));
        x174 = (x74 * x85);
        x155 = (DYI * x154);
        x156 = (x155 * x83);
        x157 = (x156 + x174 + x73);
        x142 = (x162 + (-(x163)));
        x143 = (-(x91));
        store(ConfConnectU0, stencil_idx_0_0_0_VVV, ((x120 * x137) + (x142 * x166) + (x143 * x157) + (x32 * x77) + (x38 * x39) + (x52 * x78)));
        x91 = (x37 * x54);
        x149 = (((1.0 / 2.0) * x42) + ((-1.0 / 2.0) * x43));
        x42 = (x149 * x168);
        x43 = (-(x34));
        x34 = (x173 * x43);
        x101 = (x34 + x42 + x91);
        x102 = ((2 * x141) + (-1 * DYI * x154));
        x103 = (-(x102));
        x48 = (-(x149));
        x129 = (-(x146));
        x130 = (x148 * x54);
        x131 = (x31 + (-(x126)));
        x126 = (x130 + (x103 * x48) + (x129 * x131));
        x31 = (-(x45));
        x112 = (x131 * x31);
        x113 = (x138 * x149);
        x79 = (DYI * x100);
        x56 = (((1.0 / 2.0) * x47) + (-(x79)));
        x57 = (x56 * x95);
        x58 = (x112 + x113 + x57);
        x66 = (DZI * x96);
        x67 = (((1.0 / 2.0) * x63) + (-(x66)));
        x68 = (x173 + (-2 * DZI * x36));
        vreal x184 = (-(x68));
        vreal x170 = (x180 * x43);
        vreal x185 = (x170 + (x149 * x184) + (x67 * x95));
        x170 = (x43 * x74);
        x184 = (x155 * x54);
        vreal x105 = (x149 * x47);
        vreal x108 = (x105 + x170 + x184);
        x105 = (x54 * x82);
        x54 = (x43 * x63);
        vreal x90 = (x105 + x54 + (x149 * x60));
        store(ConfConnectU1, stencil_idx_0_0_0_VVV, ((x101 * x39) + (x108 * x143) + (x126 * x78) + (x137 * x58) + (x142 * x185) + (x32 * x90)));
        x108 = (x168 * x43);
        x185 = (x37 * x85);
        x90 = (((1.0 / 2.0) * x162) + ((-1.0 / 2.0) * x163));
        x162 = (x173 * x90);
        x163 = (x108 + x162 + x185);
        vreal x239 = ((2 * x121) + (-1 * DZI * x62));
        vreal x240 = (-(x239));
        x239 = (-(x90));
        vreal x238 = (x148 * x85);
        vreal x241 = (x238 + (x131 * x94) + (x239 * x240));
        x238 = (-(x67));
        x240 = (x238 * x46);
        vreal x228 = (x180 * x90);
        vreal x231 = (x131 * x179);
        vreal x232 = (x228 + x231 + x240);
        x228 = (x138 * x43);
        x231 = ((-2 * x111) + (DZI * x128));
        vreal x234 = (-(x56));
        vreal x237 = (x228 + (x231 * x239) + (x234 * x46));
        x234 = (x155 * x85);
        vreal x225 = (x43 * x47);
        vreal x227 = (x225 + x234 + (x74 * x90));
        x225 = (x63 * x90);
        vreal x223 = (x43 * x60);
        vreal x221 = (x82 * x85);
        vreal x224 = (x221 + x223 + x225);
        store(ConfConnectU2, stencil_idx_0_0_0_VVV, ((x137 * x237) + (x142 * x232) + (x143 * x227) + (x163 * x39) + (x224 * x32) + (x241 * x78)));    
    });
    #ifdef __CUDACC__
    nvtxRangeEnd(range);
    #endif
}