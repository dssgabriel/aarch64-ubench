.text

.global clktest

clktest:
    sub        sp,  sp,  #0x30
    stp        x14, x15, [sp, #0x10]
    stp        x12, x13, [sp, #0x20]
    mov        x15, #1
    eor        x13, x13, x13
1:  subs       x0,  x0,  #20
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    add        x13, x13, x15
    b.ne       1b
    ldp        x12, x13, [sp, #0x20]
    ldp        x14, x15, [sp, #0x10]
    add        sp,  sp,  #0x30
    ret

.global lat_scalar_fadd
lat_scalar_fadd:
    ldr        s16, [x1]
    ldr        s17, [x1, #0x4]
    ldr        s18, [x1, #0x8]
1:  subs       x0,  x0,  #20
    fadd       s16, s17, s18
    fadd       s17, s18, s16
    fadd       s18, s16, s17
    fadd       s16, s17, s18
    fadd       s17, s18, s16
    fadd       s18, s16, s17
    fadd       s16, s17, s18
    fadd       s17, s18, s16
    fadd       s18, s16, s17
    fadd       s16, s17, s18
    fadd       s17, s18, s16
    fadd       s18, s16, s17
    fadd       s16, s17, s18
    fadd       s17, s18, s16
    fadd       s18, s16, s17
    fadd       s16, s17, s18
    fadd       s17, s18, s16
    fadd       s18, s16, s17
    fadd       s16, s17, s18
    fadd       s17, s18, s16
    b.ne       1b
    ret

.global lat_scalar_fmul
lat_scalar_fmul:
    ldr        s16, [x1]
    ldr        s17, [x1, #0x4]
    ldr        s18, [x1, #0x8]
1:  subs       x0,  x0,  #20
    fmul       s16, s17, s18
    fmul       s17, s18, s16
    fmul       s18, s16, s17
    fmul       s16, s17, s18
    fmul       s17, s18, s16
    fmul       s18, s16, s17
    fmul       s16, s17, s18
    fmul       s17, s18, s16
    fmul       s18, s16, s17
    fmul       s16, s17, s18
    fmul       s17, s18, s16
    fmul       s18, s16, s17
    fmul       s16, s17, s18
    fmul       s17, s18, s16
    fmul       s18, s16, s17
    fmul       s16, s17, s18
    fmul       s17, s18, s16
    fmul       s18, s16, s17
    fmul       s16, s17, s18
    fmul       s17, s18, s16
    b.ne       1b
    ret

.global lat_scalar_fmadd
lat_scalar_fmadd:
    ldr        s16, [x1]
    ldr        s17, [x1, #0x4]
    ldr        s18, [x1, #0x8]
    ldr        s19, [x1, #0xC]
1:  subs       x0,  x0,  #20
    fmadd      s16, s17, s18, s19
    fmadd      s19, s16, s17, s18
    fmadd      s18, s19, s16, s17
    fmadd      s17, s18, s19, s16
    fmadd      s16, s17, s18, s19
    fmadd      s19, s16, s17, s18
    fmadd      s18, s19, s16, s17
    fmadd      s17, s18, s19, s16
    fmadd      s16, s17, s18, s19
    fmadd      s19, s16, s17, s18
    fmadd      s18, s19, s16, s17
    fmadd      s17, s18, s19, s16
    fmadd      s16, s17, s18, s19
    fmadd      s19, s16, s17, s18
    fmadd      s18, s19, s16, s17
    fmadd      s17, s18, s19, s16
    fmadd      s16, s17, s18, s19
    fmadd      s19, s16, s17, s18
    fmadd      s18, s19, s16, s17
    fmadd      s17, s18, s19, s16
    b.ne       1b
    ret

.global lat_neon_fadd
lat_neon_fadd:
    ldr        q16,    [x1]
    ldr        q17,    [x1]
    ldr        q18,    [x1]
1:  subs       x0,     x0,     #20
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v17.4s, v18.4s, v16.4s
    fadd       v18.4s, v16.4s, v17.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v17.4s, v18.4s, v16.4s
    fadd       v18.4s, v16.4s, v17.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v17.4s, v18.4s, v16.4s
    fadd       v18.4s, v16.4s, v17.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v17.4s, v18.4s, v16.4s
    fadd       v18.4s, v16.4s, v17.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v17.4s, v18.4s, v16.4s
    fadd       v18.4s, v16.4s, v17.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v17.4s, v18.4s, v16.4s
    fadd       v18.4s, v16.4s, v17.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v17.4s, v18.4s, v16.4s
    b.ne       1b
    ret

.global lat_neon_fmul
lat_neon_fmul:
    ldr        q16,    [x1]
    ldr        q17,    [x1]
    ldr        q18,    [x1]
1:  subs       x0,     x0,     #20
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v17.4s, v18.4s, v16.4s
    fmul       v18.4s, v16.4s, v17.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v17.4s, v18.4s, v16.4s
    fmul       v18.4s, v16.4s, v17.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v17.4s, v18.4s, v16.4s
    fmul       v18.4s, v16.4s, v17.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v17.4s, v18.4s, v16.4s
    fmul       v18.4s, v16.4s, v17.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v17.4s, v18.4s, v16.4s
    fmul       v18.4s, v16.4s, v17.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v17.4s, v18.4s, v16.4s
    fmul       v18.4s, v16.4s, v17.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v17.4s, v18.4s, v16.4s
    b.ne       1b
    ret

.global lat_neon_fmla
lat_neon_fmla:
    ldr        q16,    [x1]
    ldr        q17,    [x1]
    ldr        q18,    [x1]
1:  subs       x0,     x0,     #20
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v17.4s, v18.4s, v16.4s
    fmla       v18.4s, v16.4s, v17.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v17.4s, v18.4s, v16.4s
    fmla       v18.4s, v16.4s, v17.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v17.4s, v18.4s, v16.4s
    fmla       v18.4s, v16.4s, v17.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v17.4s, v18.4s, v16.4s
    fmla       v18.4s, v16.4s, v17.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v17.4s, v18.4s, v16.4s
    fmla       v18.4s, v16.4s, v17.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v17.4s, v18.4s, v16.4s
    fmla       v18.4s, v16.4s, v17.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v17.4s, v18.4s, v16.4s
    b.ne       1b
    ret

#ifdef __ARM_FEATURE_SVE
.global lat_sve_fadd
lat_sve_fadd:
    ptrue      p0.s
    ld1w       z16.s, p0/z,  [x1]
    ld1w       z17.s, p0/z,  [x1]
    ld1w       z18.s, p0/z,  [x1]
1:  subs       x0,    x0,    #20
    fadd       z16.s, z17.s, z18.s
    fadd       z17.s, z18.s, z16.s
    fadd       z18.s, z16.s, z17.s
    fadd       z16.s, z17.s, z18.s
    fadd       z17.s, z18.s, z16.s
    fadd       z18.s, z16.s, z17.s
    fadd       z16.s, z17.s, z18.s
    fadd       z17.s, z18.s, z16.s
    fadd       z18.s, z16.s, z17.s
    fadd       z16.s, z17.s, z18.s
    fadd       z17.s, z18.s, z16.s
    fadd       z18.s, z16.s, z17.s
    fadd       z16.s, z17.s, z18.s
    fadd       z17.s, z18.s, z16.s
    fadd       z18.s, z16.s, z17.s
    fadd       z16.s, z17.s, z18.s
    fadd       z17.s, z18.s, z16.s
    fadd       z18.s, z16.s, z17.s
    fadd       z16.s, z17.s, z18.s
    fadd       z17.s, z18.s, z16.s
    b.ne       1b
    ret

.global lat_sve_fmul
lat_sve_fmul:
    ptrue      p0.s
    ld1w       z16.s, p0/z,  [x1]
    ld1w       z17.s, p0/z,  [x1]
    ld1w       z18.s, p0/z,  [x1]
1:  subs       x0,    x0,    #20
    fmul       z16.s, z17.s, z18.s
    fmul       z17.s, z18.s, z16.s
    fmul       z18.s, z16.s, z17.s
    fmul       z16.s, z17.s, z18.s
    fmul       z17.s, z18.s, z16.s
    fmul       z18.s, z16.s, z17.s
    fmul       z16.s, z17.s, z18.s
    fmul       z17.s, z18.s, z16.s
    fmul       z18.s, z16.s, z17.s
    fmul       z16.s, z17.s, z18.s
    fmul       z17.s, z18.s, z16.s
    fmul       z18.s, z16.s, z17.s
    fmul       z16.s, z17.s, z18.s
    fmul       z17.s, z18.s, z16.s
    fmul       z18.s, z16.s, z17.s
    fmul       z16.s, z17.s, z18.s
    fmul       z17.s, z18.s, z16.s
    fmul       z18.s, z16.s, z17.s
    fmul       z16.s, z17.s, z18.s
    fmul       z17.s, z18.s, z16.s
    b.ne       1b
    ret

.global lat_sve_fmla
lat_sve_fmla:
    ptrue      p0.s
    ld1w       z16.s, p0/z, [x1]
    ld1w       z17.s, p0/z, [x1]
    ld1w       z18.s, p0/z, [x1]
1:  subs       x0,    x0,   #20
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z17.s, p0/m, z18.s, z16.s
    fmla       z18.s, p0/m, z16.s, z17.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z17.s, p0/m, z18.s, z16.s
    fmla       z18.s, p0/m, z16.s, z17.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z17.s, p0/m, z18.s, z16.s
    fmla       z18.s, p0/m, z16.s, z17.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z17.s, p0/m, z18.s, z16.s
    fmla       z18.s, p0/m, z16.s, z17.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z17.s, p0/m, z18.s, z16.s
    fmla       z18.s, p0/m, z16.s, z17.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z17.s, p0/m, z18.s, z16.s
    fmla       z18.s, p0/m, z16.s, z17.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z17.s, p0/m, z18.s, z16.s
    b.ne       1b
    ret
#endif

.global thp_scalar_fadd
thp_scalar_fadd:
    ldr        s16, [x1]
    ldr        s17, [x1, #0x4]
    ldr        s18, [x1, #0x8]
    ldr        s19, [x1, #0xC]
    ldr        s20, [x1]
    ldr        s21, [x1, #0x4] 
1:  subs       x0,  x0,  #20
    fadd       s16, s17, s18
    fadd       s19, s17, s18
    fadd       s20, s17, s18
    fadd       s21, s17, s18
    fadd       s16, s17, s18
    fadd       s19, s17, s18
    fadd       s20, s17, s18
    fadd       s21, s17, s18
    fadd       s16, s17, s18
    fadd       s19, s17, s18
    fadd       s20, s17, s18
    fadd       s21, s17, s18
    fadd       s16, s17, s18
    fadd       s19, s17, s18
    fadd       s20, s17, s18
    fadd       s21, s17, s18
    fadd       s16, s17, s18
    fadd       s19, s17, s18
    fadd       s20, s17, s18
    fadd       s21, s17, s18
    b.ne       1b
    ret

.global thp_scalar_fmul
thp_scalar_fmul:
    ldr        s16, [x1]
    ldr        s17, [x1, #0x4]
    ldr        s18, [x1, #0x8]
    ldr        s19, [x1, #0xC]
    ldr        s20, [x1]
    ldr        s21, [x1, #0x4] 
1:  subs       x0,  x0,  #20
    fmul       s16, s17, s18
    fmul       s19, s17, s18
    fmul       s20, s17, s18
    fmul       s21, s17, s18
    fmul       s16, s17, s18
    fmul       s19, s17, s18
    fmul       s20, s17, s18
    fmul       s21, s17, s18
    fmul       s16, s17, s18
    fmul       s19, s17, s18
    fmul       s20, s17, s18
    fmul       s21, s17, s18
    fmul       s16, s17, s18
    fmul       s19, s17, s18
    fmul       s20, s17, s18
    fmul       s21, s17, s18
    fmul       s16, s17, s18
    fmul       s19, s17, s18
    fmul       s20, s17, s18
    fmul       s21, s17, s18
    b.ne       1b
    ret

.global thp_scalar_fmadd
thp_scalar_fmadd:
    ldr        s16, [x1]
    ldr        s17, [x1, #0x4]
    ldr        s18, [x1, #0x8]
    ldr        s19, [x1, #0xC]
    ldr        s20, [x1]
    ldr        s21, [x1, #0x4] 
    ldr        s22, [x1, #0x8] 
1:  subs       x0,  x0,  #20
    fmadd      s16, s17, s18, s19
    fmadd      s20, s17, s18, s19
    fmadd      s21, s17, s18, s19
    fmadd      s22, s17, s18, s19
    fmadd      s16, s17, s18, s19
    fmadd      s20, s17, s18, s19
    fmadd      s21, s17, s18, s19
    fmadd      s22, s17, s18, s19
    fmadd      s16, s17, s18, s19
    fmadd      s20, s17, s18, s19
    fmadd      s21, s17, s18, s19
    fmadd      s22, s17, s18, s19
    fmadd      s16, s17, s18, s19
    fmadd      s20, s17, s18, s19
    fmadd      s21, s17, s18, s19
    fmadd      s22, s17, s18, s19
    fmadd      s16, s17, s18, s19
    fmadd      s20, s17, s18, s19
    fmadd      s21, s17, s18, s19
    fmadd      s22, s17, s18, s19
    b.ne       1b
    ret

.global thp_neon_fadd
thp_neon_fadd:
    ldr        q16,    [x1]
    ldr        q17,    [x1]
    ldr        q18,    [x1]
    ldr        q19,    [x1]
    ldr        q20,    [x1]
    ldr        q21,    [x1] 
1:  subs       x0,     x0,     #20
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v19.4s, v17.4s, v18.4s
    fadd       v20.4s, v17.4s, v18.4s
    fadd       v21.4s, v17.4s, v18.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v19.4s, v17.4s, v18.4s
    fadd       v20.4s, v17.4s, v18.4s
    fadd       v21.4s, v17.4s, v18.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v19.4s, v17.4s, v18.4s
    fadd       v20.4s, v17.4s, v18.4s
    fadd       v21.4s, v17.4s, v18.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v19.4s, v17.4s, v18.4s
    fadd       v20.4s, v17.4s, v18.4s
    fadd       v21.4s, v17.4s, v18.4s
    fadd       v16.4s, v17.4s, v18.4s
    fadd       v19.4s, v17.4s, v18.4s
    fadd       v20.4s, v17.4s, v18.4s
    fadd       v21.4s, v17.4s, v18.4s
    b.ne       1b
    ret

.global thp_neon_fmul
thp_neon_fmul:
    ldr        q16,    [x1]
    ldr        q17,    [x1]
    ldr        q18,    [x1]
    ldr        q19,    [x1]
    ldr        q20,    [x1]
    ldr        q21,    [x1] 
1:  subs       x0,     x0,     #20
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v19.4s, v17.4s, v18.4s
    fmul       v20.4s, v17.4s, v18.4s
    fmul       v21.4s, v17.4s, v18.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v19.4s, v17.4s, v18.4s
    fmul       v20.4s, v17.4s, v18.4s
    fmul       v21.4s, v17.4s, v18.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v19.4s, v17.4s, v18.4s
    fmul       v20.4s, v17.4s, v18.4s
    fmul       v21.4s, v17.4s, v18.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v19.4s, v17.4s, v18.4s
    fmul       v20.4s, v17.4s, v18.4s
    fmul       v21.4s, v17.4s, v18.4s
    fmul       v16.4s, v17.4s, v18.4s
    fmul       v19.4s, v17.4s, v18.4s
    fmul       v20.4s, v17.4s, v18.4s
    fmul       v21.4s, v17.4s, v18.4s
    b.ne       1b
    ret

.global thp_neon_fmla
thp_neon_fmla:
    ldr        q16,    [x1]
    ldr        q17,    [x1]
    ldr        q18,    [x1]
    ldr        q19,    [x1]
    ldr        q20,    [x1]
    ldr        q21,    [x1] 
1:  subs       x0,     x0,     #20
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v19.4s, v17.4s, v18.4s
    fmla       v20.4s, v17.4s, v18.4s
    fmla       v21.4s, v17.4s, v18.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v19.4s, v17.4s, v18.4s
    fmla       v20.4s, v17.4s, v18.4s
    fmla       v21.4s, v17.4s, v18.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v19.4s, v17.4s, v18.4s
    fmla       v20.4s, v17.4s, v18.4s
    fmla       v21.4s, v17.4s, v18.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v19.4s, v17.4s, v18.4s
    fmla       v20.4s, v17.4s, v18.4s
    fmla       v21.4s, v17.4s, v18.4s
    fmla       v16.4s, v17.4s, v18.4s
    fmla       v19.4s, v17.4s, v18.4s
    fmla       v20.4s, v17.4s, v18.4s
    fmla       v21.4s, v17.4s, v18.4s
    b.ne       1b
    ret

.global mix_thp_neon_faddfmul
mix_thp_neon_faddfmul:
    ldr        q16,    [x1]
    ldr        q17,    [x1]
    ldr        q18,    [x1]
    ldr        q19,    [x1]
    ldr        q20,    [x1]
    ldr        q21,    [x1]
    ldr        q22,    [x1]
    ldr        q23,    [x1]
1:  subs       x0,     x0,     #20
    fmul       v16.4s, v17.4s, v18.4s
    fadd       v21.4s, v19.4s, v20.4s
    fmul       v22.4s, v17.4s, v18.4s
    fadd       v23.4s, v19.4s, v20.4s
    fmul       v16.4s, v17.4s, v18.4s
    fadd       v21.4s, v19.4s, v20.4s
    fmul       v22.4s, v17.4s, v18.4s
    fadd       v23.4s, v19.4s, v20.4s
    fmul       v16.4s, v17.4s, v18.4s
    fadd       v21.4s, v19.4s, v20.4s
    fmul       v22.4s, v17.4s, v18.4s
    fadd       v23.4s, v19.4s, v20.4s
    fmul       v16.4s, v17.4s, v18.4s
    fadd       v21.4s, v19.4s, v20.4s
    fmul       v22.4s, v17.4s, v18.4s
    fadd       v23.4s, v19.4s, v20.4s
    fmul       v16.4s, v17.4s, v18.4s
    fadd       v21.4s, v19.4s, v20.4s
    fmul       v22.4s, v17.4s, v18.4s
    fadd       v23.4s, v19.4s, v20.4s
    b.ne       1b
    ret 

#ifdef __ARM_FEATURE_SVE
.global thp_sve_fadd
thp_sve_fadd:
    ptrue      p0.s
    ld1w       z16.s, p0/z,  [x1]
    ld1w       z17.s, p0/z,  [x1]
    ld1w       z18.s, p0/z,  [x1]
    ld1w       z19.s, p0/z,  [x1]
    ld1w       z20.s, p0/z,  [x1]
    ld1w       z21.s, p0/z,  [x1]
1:  subs       x0,    x0,    #20
    fadd       z16.s, z17.s, z18.s
    fadd       z19.s, z17.s, z18.s
    fadd       z20.s, z17.s, z18.s
    fadd       z21.s, z17.s, z18.s
    fadd       z16.s, z17.s, z18.s
    fadd       z19.s, z17.s, z18.s
    fadd       z20.s, z17.s, z18.s
    fadd       z21.s, z17.s, z18.s
    fadd       z16.s, z17.s, z18.s
    fadd       z19.s, z17.s, z18.s
    fadd       z20.s, z17.s, z18.s
    fadd       z21.s, z17.s, z18.s
    fadd       z16.s, z17.s, z18.s
    fadd       z19.s, z17.s, z18.s
    fadd       z20.s, z17.s, z18.s
    fadd       z21.s, z17.s, z18.s
    fadd       z16.s, z17.s, z18.s
    fadd       z19.s, z17.s, z18.s
    fadd       z20.s, z17.s, z18.s
    fadd       z21.s, z17.s, z18.s
    b.ne       1b
    ret

.global thp_sve_fmul
thp_sve_fmul:
    ptrue      p0.s
    ld1w       z16.s, p0/z,  [x1]
    ld1w       z17.s, p0/z,  [x1]
    ld1w       z18.s, p0/z,  [x1]
    ld1w       z19.s, p0/z,  [x1]
    ld1w       z20.s, p0/z,  [x1]
    ld1w       z21.s, p0/z,  [x1]
1:  subs       x0,    x0,    #20
    fmul       z16.s, z17.s, z18.s
    fmul       z19.s, z17.s, z18.s
    fmul       z20.s, z17.s, z18.s
    fmul       z21.s, z17.s, z18.s
    fmul       z16.s, z17.s, z18.s
    fmul       z19.s, z17.s, z18.s
    fmul       z20.s, z17.s, z18.s
    fmul       z21.s, z17.s, z18.s
    fmul       z16.s, z17.s, z18.s
    fmul       z19.s, z17.s, z18.s
    fmul       z20.s, z17.s, z18.s
    fmul       z21.s, z17.s, z18.s
    fmul       z16.s, z17.s, z18.s
    fmul       z19.s, z17.s, z18.s
    fmul       z20.s, z17.s, z18.s
    fmul       z21.s, z17.s, z18.s
    fmul       z16.s, z17.s, z18.s
    fmul       z19.s, z17.s, z18.s
    fmul       z20.s, z17.s, z18.s
    fmul       z21.s, z17.s, z18.s
    b.ne       1b
    ret

.global thp_sve_fmla
thp_sve_fmla:
    ptrue      p0.s
    ld1w       z16.s, p0/z, [x1]
    ld1w       z17.s, p0/z, [x1]
    ld1w       z18.s, p0/z, [x1]
    ld1w       z19.s, p0/z, [x1]
    ld1w       z20.s, p0/z, [x1]
    ld1w       z21.s, p0/z, [x1]
1:  subs       x0,    x0,   #20
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z19.s, p0/m, z17.s, z18.s
    fmla       z20.s, p0/m, z17.s, z18.s
    fmla       z21.s, p0/m, z17.s, z18.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z19.s, p0/m, z17.s, z18.s
    fmla       z20.s, p0/m, z17.s, z18.s
    fmla       z21.s, p0/m, z17.s, z18.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z19.s, p0/m, z17.s, z18.s
    fmla       z20.s, p0/m, z17.s, z18.s
    fmla       z21.s, p0/m, z17.s, z18.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z19.s, p0/m, z17.s, z18.s
    fmla       z20.s, p0/m, z17.s, z18.s
    fmla       z21.s, p0/m, z17.s, z18.s
    fmla       z16.s, p0/m, z17.s, z18.s
    fmla       z19.s, p0/m, z17.s, z18.s
    fmla       z20.s, p0/m, z17.s, z18.s
    fmla       z21.s, p0/m, z17.s, z18.s
    b.ne       1b
    ret

.global mix_thp_sve_faddfmul
mix_thp_sve_faddfmul:
    ptrue      p0.s
    ld1w       z16.s, p0/z,  [x1]
    ld1w       z17.s, p0/z,  [x1]
    ld1w       z18.s, p0/z,  [x1]
    ld1w       z19.s, p0/z,  [x1]
    ld1w       z20.s, p0/z,  [x1]
    ld1w       z21.s, p0/z,  [x1]
    ld1w       z22.s, p0/z,  [x1]
    ld1w       z23.s, p0/z,  [x1]
1:  subs       x0,    x0,    #20
    fmul       z16.s, z17.s, z18.s
    fadd       z21.s, z19.s, z20.s
    fmul       z22.s, z17.s, z18.s
    fadd       z23.s, z19.s, z20.s
    fmul       z16.s, z17.s, z18.s
    fadd       z21.s, z19.s, z20.s
    fmul       z22.s, z17.s, z18.s
    fadd       z23.s, z19.s, z20.s
    fmul       z16.s, z17.s, z18.s
    fadd       z21.s, z19.s, z20.s
    fmul       z22.s, z17.s, z18.s
    fadd       z23.s, z19.s, z20.s
    fmul       z16.s, z17.s, z18.s
    fadd       z21.s, z19.s, z20.s
    fmul       z22.s, z17.s, z18.s
    fadd       z23.s, z19.s, z20.s
    fmul       z16.s, z17.s, z18.s
    fadd       z21.s, z19.s, z20.s
    fmul       z22.s, z17.s, z18.s
    fadd       z23.s, z19.s, z20.s
    b.ne       1b
    ret 
#endif
