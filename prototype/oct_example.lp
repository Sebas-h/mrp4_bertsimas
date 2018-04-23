\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: - node_1_applies_split + node_1_considers_var_0
   + node_1_considers_var_1 + node_1_considers_var_2
   + node_1_considers_var_3 = 0
 R1: split_value_node_1 - node_1_applies_split <= 0
 R2: - node_2_applies_split + node_2_considers_var_0
   + node_2_considers_var_1 + node_2_considers_var_2
   + node_2_considers_var_3 = 0
 R3: split_value_node_2 - node_2_applies_split <= 0
 R4: - node_1_applies_split + node_2_applies_split <= 0
 R5: - node_3_applies_split + node_3_considers_var_0
   + node_3_considers_var_1 + node_3_considers_var_2
   + node_3_considers_var_3 = 0
 R6: split_value_node_3 - node_3_applies_split <= 0
 R7: - node_1_applies_split + node_3_applies_split <= 0
 R8: - node_4_applies_split + node_4_considers_var_0
   + node_4_considers_var_1 + node_4_considers_var_2
   + node_4_considers_var_3 = 0
 R9: split_value_node_4 - node_4_applies_split <= 0
 R10: - node_2_applies_split + node_4_applies_split <= 0
 R11: - node_5_applies_split + node_5_considers_var_0
   + node_5_considers_var_1 + node_5_considers_var_2
   + node_5_considers_var_3 = 0
 R12: split_value_node_5 - node_5_applies_split <= 0
 R13: - node_2_applies_split + node_5_applies_split <= 0
 R14: - node_6_applies_split + node_6_considers_var_0
   + node_6_considers_var_1 + node_6_considers_var_2
   + node_6_considers_var_3 = 0
 R15: split_value_node_6 - node_6_applies_split <= 0
 R16: - node_3_applies_split + node_6_applies_split <= 0
 R17: - node_7_applies_split + node_7_considers_var_0
   + node_7_considers_var_1 + node_7_considers_var_2
   + node_7_considers_var_3 = 0
 R18: split_value_node_7 - node_7_applies_split <= 0
 R19: - node_3_applies_split + node_7_applies_split <= 0
 R20: - node_8_contains_any_point + x0_is_in_node_8 <= 0
 R21: - node_8_contains_any_point + x1_is_in_node_8 <= 0
 R22: - node_8_contains_any_point + x2_is_in_node_8 <= 0
 R23: - node_8_contains_any_point + x3_is_in_node_8 <= 0
 R24: - node_8_contains_any_point + x4_is_in_node_8 <= 0
 R25: - node_8_contains_any_point + x5_is_in_node_8 <= 0
 R26: - node_8_contains_any_point + x6_is_in_node_8 <= 0
 R27: - node_8_contains_any_point + x7_is_in_node_8 <= 0
 R28: - node_8_contains_any_point + x8_is_in_node_8 <= 0
 R29: - node_8_contains_any_point + x9_is_in_node_8 <= 0
 R30: - node_8_contains_any_point + x10_is_in_node_8 <= 0
 R31: - node_8_contains_any_point + x11_is_in_node_8 <= 0
 R32: - node_8_contains_any_point + x12_is_in_node_8 <= 0
 R33: - node_8_contains_any_point + x13_is_in_node_8 <= 0
 R34: - node_8_contains_any_point + x14_is_in_node_8 <= 0
 R35: - node_8_contains_any_point + x15_is_in_node_8 <= 0
 R36: - node_8_contains_any_point + x16_is_in_node_8 <= 0
 R37: - node_8_contains_any_point + x17_is_in_node_8 <= 0
 R38: - node_8_contains_any_point + x18_is_in_node_8 <= 0
 R39: - node_8_contains_any_point + x19_is_in_node_8 <= 0
 R40: - node_8_contains_any_point + x20_is_in_node_8 <= 0
 R41: - node_8_contains_any_point + x21_is_in_node_8 <= 0
 R42: - node_8_contains_any_point + x22_is_in_node_8 <= 0
 R43: - node_8_contains_any_point + x23_is_in_node_8 <= 0
 R44: - node_8_contains_any_point + x24_is_in_node_8 <= 0
 R45: - node_8_contains_any_point + x25_is_in_node_8 <= 0
 R46: - node_8_contains_any_point + x26_is_in_node_8 <= 0
 R47: - node_8_contains_any_point + x27_is_in_node_8 <= 0
 R48: - node_8_contains_any_point + x28_is_in_node_8 <= 0
 R49: - node_8_contains_any_point + x29_is_in_node_8 <= 0
 R50: - node_8_contains_any_point + x30_is_in_node_8 <= 0
 R51: - node_8_contains_any_point + x31_is_in_node_8 <= 0
 R52: - node_8_contains_any_point + x32_is_in_node_8 <= 0
 R53: - node_8_contains_any_point + x33_is_in_node_8 <= 0
 R54: - node_8_contains_any_point + x34_is_in_node_8 <= 0
 R55: - node_8_contains_any_point + x35_is_in_node_8 <= 0
 R56: - node_8_contains_any_point + x36_is_in_node_8 <= 0
 R57: - node_8_contains_any_point + x37_is_in_node_8 <= 0
 R58: - node_8_contains_any_point + x38_is_in_node_8 <= 0
 R59: - node_8_contains_any_point + x39_is_in_node_8 <= 0
 R60: - node_8_contains_any_point + x40_is_in_node_8 <= 0
 R61: - node_8_contains_any_point + x41_is_in_node_8 <= 0
 R62: - node_8_contains_any_point + x42_is_in_node_8 <= 0
 R63: - node_8_contains_any_point + x43_is_in_node_8 <= 0
 R64: - node_8_contains_any_point + x44_is_in_node_8 <= 0
 R65: - node_8_contains_any_point + x45_is_in_node_8 <= 0
 R66: - node_8_contains_any_point + x46_is_in_node_8 <= 0
 R67: - node_8_contains_any_point + x47_is_in_node_8 <= 0
 R68: - node_8_contains_any_point + x48_is_in_node_8 <= 0
 R69: - node_8_contains_any_point + x49_is_in_node_8 <= 0
 R70: - node_8_contains_any_point + x50_is_in_node_8 <= 0
 R71: - node_8_contains_any_point + x51_is_in_node_8 <= 0
 R72: - node_8_contains_any_point + x52_is_in_node_8 <= 0
 R73: - node_8_contains_any_point + x53_is_in_node_8 <= 0
 R74: - node_8_contains_any_point + x54_is_in_node_8 <= 0
 R75: - node_8_contains_any_point + x55_is_in_node_8 <= 0
 R76: - node_8_contains_any_point + x56_is_in_node_8 <= 0
 R77: - node_8_contains_any_point + x57_is_in_node_8 <= 0
 R78: - node_8_contains_any_point + x58_is_in_node_8 <= 0
 R79: - node_8_contains_any_point + x59_is_in_node_8 <= 0
 R80: - node_8_contains_any_point + x60_is_in_node_8 <= 0
 R81: - node_8_contains_any_point + x61_is_in_node_8 <= 0
 R82: - node_8_contains_any_point + x62_is_in_node_8 <= 0
 R83: - node_8_contains_any_point + x63_is_in_node_8 <= 0
 R84: - node_8_contains_any_point + x64_is_in_node_8 <= 0
 R85: - node_8_contains_any_point + x65_is_in_node_8 <= 0
 R86: - node_8_contains_any_point + x66_is_in_node_8 <= 0
 R87: - node_8_contains_any_point + x67_is_in_node_8 <= 0
 R88: - node_8_contains_any_point + x68_is_in_node_8 <= 0
 R89: - node_8_contains_any_point + x69_is_in_node_8 <= 0
 R90: - node_8_contains_any_point + x70_is_in_node_8 <= 0
 R91: - node_8_contains_any_point + x71_is_in_node_8 <= 0
 R92: - node_8_contains_any_point + x72_is_in_node_8 <= 0
 R93: - node_8_contains_any_point + x73_is_in_node_8 <= 0
 R94: - node_8_contains_any_point + x74_is_in_node_8 <= 0
 R95: - node_8_contains_any_point + x75_is_in_node_8 <= 0
 R96: - node_8_contains_any_point + x76_is_in_node_8 <= 0
 R97: - node_8_contains_any_point + x77_is_in_node_8 <= 0
 R98: - node_8_contains_any_point + x78_is_in_node_8 <= 0
 R99: - node_8_contains_any_point + x79_is_in_node_8 <= 0
 R100: - node_8_contains_any_point + x80_is_in_node_8 <= 0
 R101: - node_8_contains_any_point + x81_is_in_node_8 <= 0
 R102: - node_8_contains_any_point + x82_is_in_node_8 <= 0
 R103: - node_8_contains_any_point + x83_is_in_node_8 <= 0
 R104: - node_8_contains_any_point + x84_is_in_node_8 <= 0
 R105: - node_8_contains_any_point + x85_is_in_node_8 <= 0
 R106: - node_8_contains_any_point + x86_is_in_node_8 <= 0
 R107: - node_8_contains_any_point + x87_is_in_node_8 <= 0
 R108: - node_8_contains_any_point + x88_is_in_node_8 <= 0
 R109: - node_8_contains_any_point + x89_is_in_node_8 <= 0
 R110: - node_8_contains_any_point + x90_is_in_node_8 <= 0
 R111: - node_8_contains_any_point + x91_is_in_node_8 <= 0
 R112: - node_8_contains_any_point + x92_is_in_node_8 <= 0
 R113: - node_8_contains_any_point + x93_is_in_node_8 <= 0
 R114: - node_8_contains_any_point + x94_is_in_node_8 <= 0
 R115: - node_8_contains_any_point + x95_is_in_node_8 <= 0
 R116: - node_8_contains_any_point + x96_is_in_node_8 <= 0
 R117: - node_8_contains_any_point + x97_is_in_node_8 <= 0
 R118: - node_8_contains_any_point + x98_is_in_node_8 <= 0
 R119: - node_8_contains_any_point + x99_is_in_node_8 <= 0
 R120: - node_8_contains_any_point + x100_is_in_node_8 <= 0
 R121: - node_8_contains_any_point + x101_is_in_node_8 <= 0
 R122: - node_8_contains_any_point + x102_is_in_node_8 <= 0
 R123: - node_8_contains_any_point + x103_is_in_node_8 <= 0
 R124: - node_8_contains_any_point + x104_is_in_node_8 <= 0
 R125: - node_8_contains_any_point + x105_is_in_node_8 <= 0
 R126: - node_8_contains_any_point + x106_is_in_node_8 <= 0
 R127: - node_8_contains_any_point + x107_is_in_node_8 <= 0
 R128: - node_8_contains_any_point + x108_is_in_node_8 <= 0
 R129: - node_8_contains_any_point + x109_is_in_node_8 <= 0
 R130: - node_8_contains_any_point + x110_is_in_node_8 <= 0
 R131: - node_8_contains_any_point + x111_is_in_node_8 <= 0
 R132: - node_8_contains_any_point + x112_is_in_node_8 <= 0
 R133: - node_8_contains_any_point + x113_is_in_node_8 <= 0
 R134: - node_8_contains_any_point + x114_is_in_node_8 <= 0
 R135: - node_8_contains_any_point + x115_is_in_node_8 <= 0
 R136: - node_8_contains_any_point + x116_is_in_node_8 <= 0
 R137: - node_8_contains_any_point + x117_is_in_node_8 <= 0
 R138: - node_8_contains_any_point + x118_is_in_node_8 <= 0
 R139: - node_8_contains_any_point + x119_is_in_node_8 <= 0
 R140: - node_8_contains_any_point + x120_is_in_node_8 <= 0
 R141: - node_8_contains_any_point + x121_is_in_node_8 <= 0
 R142: - node_8_contains_any_point + x122_is_in_node_8 <= 0
 R143: - node_8_contains_any_point + x123_is_in_node_8 <= 0
 R144: - node_8_contains_any_point + x124_is_in_node_8 <= 0
 R145: - node_8_contains_any_point + x125_is_in_node_8 <= 0
 R146: - node_8_contains_any_point + x126_is_in_node_8 <= 0
 R147: - node_8_contains_any_point + x127_is_in_node_8 <= 0
 R148: - node_8_contains_any_point + x128_is_in_node_8 <= 0
 R149: - node_8_contains_any_point + x129_is_in_node_8 <= 0
 R150: - node_8_contains_any_point + x130_is_in_node_8 <= 0
 R151: - node_8_contains_any_point + x131_is_in_node_8 <= 0
 R152: - node_8_contains_any_point + x132_is_in_node_8 <= 0
 R153: - node_8_contains_any_point + x133_is_in_node_8 <= 0
 R154: - node_8_contains_any_point + x134_is_in_node_8 <= 0
 R155: - node_8_contains_any_point + x135_is_in_node_8 <= 0
 R156: - node_8_contains_any_point + x136_is_in_node_8 <= 0
 R157: - node_8_contains_any_point + x137_is_in_node_8 <= 0
 R158: - node_8_contains_any_point + x138_is_in_node_8 <= 0
 R159: - node_8_contains_any_point + x139_is_in_node_8 <= 0
 R160: - node_8_contains_any_point + x140_is_in_node_8 <= 0
 R161: - node_8_contains_any_point + x141_is_in_node_8 <= 0
 R162: - node_8_contains_any_point + x142_is_in_node_8 <= 0
 R163: - node_8_contains_any_point + x143_is_in_node_8 <= 0
 R164: - node_8_contains_any_point + x144_is_in_node_8 <= 0
 R165: - node_8_contains_any_point + x145_is_in_node_8 <= 0
 R166: - node_8_contains_any_point + x146_is_in_node_8 <= 0
 R167: - node_8_contains_any_point + x147_is_in_node_8 <= 0
 R168: - node_8_contains_any_point + x148_is_in_node_8 <= 0
 R169: - node_8_contains_any_point + x149_is_in_node_8 <= 0
 R170: - 7 node_8_contains_any_point + x0_is_in_node_8 + x1_is_in_node_8
   + x2_is_in_node_8 + x3_is_in_node_8 + x4_is_in_node_8 + x5_is_in_node_8
   + x6_is_in_node_8 + x7_is_in_node_8 + x8_is_in_node_8 + x9_is_in_node_8
   + x10_is_in_node_8 + x11_is_in_node_8 + x12_is_in_node_8
   + x13_is_in_node_8 + x14_is_in_node_8 + x15_is_in_node_8
   + x16_is_in_node_8 + x17_is_in_node_8 + x18_is_in_node_8
   + x19_is_in_node_8 + x20_is_in_node_8 + x21_is_in_node_8
   + x22_is_in_node_8 + x23_is_in_node_8 + x24_is_in_node_8
   + x25_is_in_node_8 + x26_is_in_node_8 + x27_is_in_node_8
   + x28_is_in_node_8 + x29_is_in_node_8 + x30_is_in_node_8
   + x31_is_in_node_8 + x32_is_in_node_8 + x33_is_in_node_8
   + x34_is_in_node_8 + x35_is_in_node_8 + x36_is_in_node_8
   + x37_is_in_node_8 + x38_is_in_node_8 + x39_is_in_node_8
   + x40_is_in_node_8 + x41_is_in_node_8 + x42_is_in_node_8
   + x43_is_in_node_8 + x44_is_in_node_8 + x45_is_in_node_8
   + x46_is_in_node_8 + x47_is_in_node_8 + x48_is_in_node_8
   + x49_is_in_node_8 + x50_is_in_node_8 + x51_is_in_node_8
   + x52_is_in_node_8 + x53_is_in_node_8 + x54_is_in_node_8
   + x55_is_in_node_8 + x56_is_in_node_8 + x57_is_in_node_8
   + x58_is_in_node_8 + x59_is_in_node_8 + x60_is_in_node_8
   + x61_is_in_node_8 + x62_is_in_node_8 + x63_is_in_node_8
   + x64_is_in_node_8 + x65_is_in_node_8 + x66_is_in_node_8
   + x67_is_in_node_8 + x68_is_in_node_8 + x69_is_in_node_8
   + x70_is_in_node_8 + x71_is_in_node_8 + x72_is_in_node_8
   + x73_is_in_node_8 + x74_is_in_node_8 + x75_is_in_node_8
   + x76_is_in_node_8 + x77_is_in_node_8 + x78_is_in_node_8
   + x79_is_in_node_8 + x80_is_in_node_8 + x81_is_in_node_8
   + x82_is_in_node_8 + x83_is_in_node_8 + x84_is_in_node_8
   + x85_is_in_node_8 + x86_is_in_node_8 + x87_is_in_node_8
   + x88_is_in_node_8 + x89_is_in_node_8 + x90_is_in_node_8
   + x91_is_in_node_8 + x92_is_in_node_8 + x93_is_in_node_8
   + x94_is_in_node_8 + x95_is_in_node_8 + x96_is_in_node_8
   + x97_is_in_node_8 + x98_is_in_node_8 + x99_is_in_node_8
   + x100_is_in_node_8 + x101_is_in_node_8 + x102_is_in_node_8
   + x103_is_in_node_8 + x104_is_in_node_8 + x105_is_in_node_8
   + x106_is_in_node_8 + x107_is_in_node_8 + x108_is_in_node_8
   + x109_is_in_node_8 + x110_is_in_node_8 + x111_is_in_node_8
   + x112_is_in_node_8 + x113_is_in_node_8 + x114_is_in_node_8
   + x115_is_in_node_8 + x116_is_in_node_8 + x117_is_in_node_8
   + x118_is_in_node_8 + x119_is_in_node_8 + x120_is_in_node_8
   + x121_is_in_node_8 + x122_is_in_node_8 + x123_is_in_node_8
   + x124_is_in_node_8 + x125_is_in_node_8 + x126_is_in_node_8
   + x127_is_in_node_8 + x128_is_in_node_8 + x129_is_in_node_8
   + x130_is_in_node_8 + x131_is_in_node_8 + x132_is_in_node_8
   + x133_is_in_node_8 + x134_is_in_node_8 + x135_is_in_node_8
   + x136_is_in_node_8 + x137_is_in_node_8 + x138_is_in_node_8
   + x139_is_in_node_8 + x140_is_in_node_8 + x141_is_in_node_8
   + x142_is_in_node_8 + x143_is_in_node_8 + x144_is_in_node_8
   + x145_is_in_node_8 + x146_is_in_node_8 + x147_is_in_node_8
   + x148_is_in_node_8 + x149_is_in_node_8 >= 0
 R171: x0_is_in_node_8 + x1_is_in_node_8 + x2_is_in_node_8
   + x3_is_in_node_8 + x4_is_in_node_8 + x5_is_in_node_8 + x6_is_in_node_8
   + x7_is_in_node_8 + x8_is_in_node_8 + x9_is_in_node_8 + x10_is_in_node_8
   + x11_is_in_node_8 + x12_is_in_node_8 + x13_is_in_node_8
   + x14_is_in_node_8 + x15_is_in_node_8 + x16_is_in_node_8
   + x17_is_in_node_8 + x18_is_in_node_8 + x19_is_in_node_8
   + x20_is_in_node_8 + x21_is_in_node_8 + x22_is_in_node_8
   + x23_is_in_node_8 + x24_is_in_node_8 + x25_is_in_node_8
   + x26_is_in_node_8 + x27_is_in_node_8 + x28_is_in_node_8
   + x29_is_in_node_8 + x30_is_in_node_8 + x31_is_in_node_8
   + x32_is_in_node_8 + x33_is_in_node_8 + x34_is_in_node_8
   + x35_is_in_node_8 + x36_is_in_node_8 + x37_is_in_node_8
   + x38_is_in_node_8 + x39_is_in_node_8 + x40_is_in_node_8
   + x41_is_in_node_8 + x42_is_in_node_8 + x43_is_in_node_8
   + x44_is_in_node_8 + x45_is_in_node_8 + x46_is_in_node_8
   + x47_is_in_node_8 + x48_is_in_node_8 + x49_is_in_node_8
   + x50_is_in_node_8 + x51_is_in_node_8 + x52_is_in_node_8
   + x53_is_in_node_8 + x54_is_in_node_8 + x55_is_in_node_8
   + x56_is_in_node_8 + x57_is_in_node_8 + x58_is_in_node_8
   + x59_is_in_node_8 + x60_is_in_node_8 + x61_is_in_node_8
   + x62_is_in_node_8 + x63_is_in_node_8 + x64_is_in_node_8
   + x65_is_in_node_8 + x66_is_in_node_8 + x67_is_in_node_8
   + x68_is_in_node_8 + x69_is_in_node_8 + x70_is_in_node_8
   + x71_is_in_node_8 + x72_is_in_node_8 + x73_is_in_node_8
   + x74_is_in_node_8 + x75_is_in_node_8 + x76_is_in_node_8
   + x77_is_in_node_8 + x78_is_in_node_8 + x79_is_in_node_8
   + x80_is_in_node_8 + x81_is_in_node_8 + x82_is_in_node_8
   + x83_is_in_node_8 + x84_is_in_node_8 + x85_is_in_node_8
   + x86_is_in_node_8 + x87_is_in_node_8 + x88_is_in_node_8
   + x89_is_in_node_8 + x90_is_in_node_8 + x91_is_in_node_8
   + x92_is_in_node_8 + x93_is_in_node_8 + x94_is_in_node_8
   + x95_is_in_node_8 + x96_is_in_node_8 + x97_is_in_node_8
   + x98_is_in_node_8 + x99_is_in_node_8 + x100_is_in_node_8
   + x101_is_in_node_8 + x102_is_in_node_8 + x103_is_in_node_8
   + x104_is_in_node_8 + x105_is_in_node_8 + x106_is_in_node_8
   + x107_is_in_node_8 + x108_is_in_node_8 + x109_is_in_node_8
   + x110_is_in_node_8 + x111_is_in_node_8 + x112_is_in_node_8
   + x113_is_in_node_8 + x114_is_in_node_8 + x115_is_in_node_8
   + x116_is_in_node_8 + x117_is_in_node_8 + x118_is_in_node_8
   + x119_is_in_node_8 + x120_is_in_node_8 + x121_is_in_node_8
   + x122_is_in_node_8 + x123_is_in_node_8 + x124_is_in_node_8
   + x125_is_in_node_8 + x126_is_in_node_8 + x127_is_in_node_8
   + x128_is_in_node_8 + x129_is_in_node_8 + x130_is_in_node_8
   + x131_is_in_node_8 + x132_is_in_node_8 + x133_is_in_node_8
   + x134_is_in_node_8 + x135_is_in_node_8 + x136_is_in_node_8
   + x137_is_in_node_8 + x138_is_in_node_8 + x139_is_in_node_8
   + x140_is_in_node_8 + x141_is_in_node_8 + x142_is_in_node_8
   + x143_is_in_node_8 + x144_is_in_node_8 + x145_is_in_node_8
   + x146_is_in_node_8 + x147_is_in_node_8 + x148_is_in_node_8
   + x149_is_in_node_8 = 1
 R172: - node_9_contains_any_point + x0_is_in_node_9 <= 0
 R173: - node_9_contains_any_point + x1_is_in_node_9 <= 0
 R174: - node_9_contains_any_point + x2_is_in_node_9 <= 0
 R175: - node_9_contains_any_point + x3_is_in_node_9 <= 0
 R176: - node_9_contains_any_point + x4_is_in_node_9 <= 0
 R177: - node_9_contains_any_point + x5_is_in_node_9 <= 0
 R178: - node_9_contains_any_point + x6_is_in_node_9 <= 0
 R179: - node_9_contains_any_point + x7_is_in_node_9 <= 0
 R180: - node_9_contains_any_point + x8_is_in_node_9 <= 0
 R181: - node_9_contains_any_point + x9_is_in_node_9 <= 0
 R182: - node_9_contains_any_point + x10_is_in_node_9 <= 0
 R183: - node_9_contains_any_point + x11_is_in_node_9 <= 0
 R184: - node_9_contains_any_point + x12_is_in_node_9 <= 0
 R185: - node_9_contains_any_point + x13_is_in_node_9 <= 0
 R186: - node_9_contains_any_point + x14_is_in_node_9 <= 0
 R187: - node_9_contains_any_point + x15_is_in_node_9 <= 0
 R188: - node_9_contains_any_point + x16_is_in_node_9 <= 0
 R189: - node_9_contains_any_point + x17_is_in_node_9 <= 0
 R190: - node_9_contains_any_point + x18_is_in_node_9 <= 0
 R191: - node_9_contains_any_point + x19_is_in_node_9 <= 0
 R192: - node_9_contains_any_point + x20_is_in_node_9 <= 0
 R193: - node_9_contains_any_point + x21_is_in_node_9 <= 0
 R194: - node_9_contains_any_point + x22_is_in_node_9 <= 0
 R195: - node_9_contains_any_point + x23_is_in_node_9 <= 0
 R196: - node_9_contains_any_point + x24_is_in_node_9 <= 0
 R197: - node_9_contains_any_point + x25_is_in_node_9 <= 0
 R198: - node_9_contains_any_point + x26_is_in_node_9 <= 0
 R199: - node_9_contains_any_point + x27_is_in_node_9 <= 0
 R200: - node_9_contains_any_point + x28_is_in_node_9 <= 0
 R201: - node_9_contains_any_point + x29_is_in_node_9 <= 0
 R202: - node_9_contains_any_point + x30_is_in_node_9 <= 0
 R203: - node_9_contains_any_point + x31_is_in_node_9 <= 0
 R204: - node_9_contains_any_point + x32_is_in_node_9 <= 0
 R205: - node_9_contains_any_point + x33_is_in_node_9 <= 0
 R206: - node_9_contains_any_point + x34_is_in_node_9 <= 0
 R207: - node_9_contains_any_point + x35_is_in_node_9 <= 0
 R208: - node_9_contains_any_point + x36_is_in_node_9 <= 0
 R209: - node_9_contains_any_point + x37_is_in_node_9 <= 0
 R210: - node_9_contains_any_point + x38_is_in_node_9 <= 0
 R211: - node_9_contains_any_point + x39_is_in_node_9 <= 0
 R212: - node_9_contains_any_point + x40_is_in_node_9 <= 0
 R213: - node_9_contains_any_point + x41_is_in_node_9 <= 0
 R214: - node_9_contains_any_point + x42_is_in_node_9 <= 0
 R215: - node_9_contains_any_point + x43_is_in_node_9 <= 0
 R216: - node_9_contains_any_point + x44_is_in_node_9 <= 0
 R217: - node_9_contains_any_point + x45_is_in_node_9 <= 0
 R218: - node_9_contains_any_point + x46_is_in_node_9 <= 0
 R219: - node_9_contains_any_point + x47_is_in_node_9 <= 0
 R220: - node_9_contains_any_point + x48_is_in_node_9 <= 0
 R221: - node_9_contains_any_point + x49_is_in_node_9 <= 0
 R222: - node_9_contains_any_point + x50_is_in_node_9 <= 0
 R223: - node_9_contains_any_point + x51_is_in_node_9 <= 0
 R224: - node_9_contains_any_point + x52_is_in_node_9 <= 0
 R225: - node_9_contains_any_point + x53_is_in_node_9 <= 0
 R226: - node_9_contains_any_point + x54_is_in_node_9 <= 0
 R227: - node_9_contains_any_point + x55_is_in_node_9 <= 0
 R228: - node_9_contains_any_point + x56_is_in_node_9 <= 0
 R229: - node_9_contains_any_point + x57_is_in_node_9 <= 0
 R230: - node_9_contains_any_point + x58_is_in_node_9 <= 0
 R231: - node_9_contains_any_point + x59_is_in_node_9 <= 0
 R232: - node_9_contains_any_point + x60_is_in_node_9 <= 0
 R233: - node_9_contains_any_point + x61_is_in_node_9 <= 0
 R234: - node_9_contains_any_point + x62_is_in_node_9 <= 0
 R235: - node_9_contains_any_point + x63_is_in_node_9 <= 0
 R236: - node_9_contains_any_point + x64_is_in_node_9 <= 0
 R237: - node_9_contains_any_point + x65_is_in_node_9 <= 0
 R238: - node_9_contains_any_point + x66_is_in_node_9 <= 0
 R239: - node_9_contains_any_point + x67_is_in_node_9 <= 0
 R240: - node_9_contains_any_point + x68_is_in_node_9 <= 0
 R241: - node_9_contains_any_point + x69_is_in_node_9 <= 0
 R242: - node_9_contains_any_point + x70_is_in_node_9 <= 0
 R243: - node_9_contains_any_point + x71_is_in_node_9 <= 0
 R244: - node_9_contains_any_point + x72_is_in_node_9 <= 0
 R245: - node_9_contains_any_point + x73_is_in_node_9 <= 0
 R246: - node_9_contains_any_point + x74_is_in_node_9 <= 0
 R247: - node_9_contains_any_point + x75_is_in_node_9 <= 0
 R248: - node_9_contains_any_point + x76_is_in_node_9 <= 0
 R249: - node_9_contains_any_point + x77_is_in_node_9 <= 0
 R250: - node_9_contains_any_point + x78_is_in_node_9 <= 0
 R251: - node_9_contains_any_point + x79_is_in_node_9 <= 0
 R252: - node_9_contains_any_point + x80_is_in_node_9 <= 0
 R253: - node_9_contains_any_point + x81_is_in_node_9 <= 0
 R254: - node_9_contains_any_point + x82_is_in_node_9 <= 0
 R255: - node_9_contains_any_point + x83_is_in_node_9 <= 0
 R256: - node_9_contains_any_point + x84_is_in_node_9 <= 0
 R257: - node_9_contains_any_point + x85_is_in_node_9 <= 0
 R258: - node_9_contains_any_point + x86_is_in_node_9 <= 0
 R259: - node_9_contains_any_point + x87_is_in_node_9 <= 0
 R260: - node_9_contains_any_point + x88_is_in_node_9 <= 0
 R261: - node_9_contains_any_point + x89_is_in_node_9 <= 0
 R262: - node_9_contains_any_point + x90_is_in_node_9 <= 0
 R263: - node_9_contains_any_point + x91_is_in_node_9 <= 0
 R264: - node_9_contains_any_point + x92_is_in_node_9 <= 0
 R265: - node_9_contains_any_point + x93_is_in_node_9 <= 0
 R266: - node_9_contains_any_point + x94_is_in_node_9 <= 0
 R267: - node_9_contains_any_point + x95_is_in_node_9 <= 0
 R268: - node_9_contains_any_point + x96_is_in_node_9 <= 0
 R269: - node_9_contains_any_point + x97_is_in_node_9 <= 0
 R270: - node_9_contains_any_point + x98_is_in_node_9 <= 0
 R271: - node_9_contains_any_point + x99_is_in_node_9 <= 0
 R272: - node_9_contains_any_point + x100_is_in_node_9 <= 0
 R273: - node_9_contains_any_point + x101_is_in_node_9 <= 0
 R274: - node_9_contains_any_point + x102_is_in_node_9 <= 0
 R275: - node_9_contains_any_point + x103_is_in_node_9 <= 0
 R276: - node_9_contains_any_point + x104_is_in_node_9 <= 0
 R277: - node_9_contains_any_point + x105_is_in_node_9 <= 0
 R278: - node_9_contains_any_point + x106_is_in_node_9 <= 0
 R279: - node_9_contains_any_point + x107_is_in_node_9 <= 0
 R280: - node_9_contains_any_point + x108_is_in_node_9 <= 0
 R281: - node_9_contains_any_point + x109_is_in_node_9 <= 0
 R282: - node_9_contains_any_point + x110_is_in_node_9 <= 0
 R283: - node_9_contains_any_point + x111_is_in_node_9 <= 0
 R284: - node_9_contains_any_point + x112_is_in_node_9 <= 0
 R285: - node_9_contains_any_point + x113_is_in_node_9 <= 0
 R286: - node_9_contains_any_point + x114_is_in_node_9 <= 0
 R287: - node_9_contains_any_point + x115_is_in_node_9 <= 0
 R288: - node_9_contains_any_point + x116_is_in_node_9 <= 0
 R289: - node_9_contains_any_point + x117_is_in_node_9 <= 0
 R290: - node_9_contains_any_point + x118_is_in_node_9 <= 0
 R291: - node_9_contains_any_point + x119_is_in_node_9 <= 0
 R292: - node_9_contains_any_point + x120_is_in_node_9 <= 0
 R293: - node_9_contains_any_point + x121_is_in_node_9 <= 0
 R294: - node_9_contains_any_point + x122_is_in_node_9 <= 0
 R295: - node_9_contains_any_point + x123_is_in_node_9 <= 0
 R296: - node_9_contains_any_point + x124_is_in_node_9 <= 0
 R297: - node_9_contains_any_point + x125_is_in_node_9 <= 0
 R298: - node_9_contains_any_point + x126_is_in_node_9 <= 0
 R299: - node_9_contains_any_point + x127_is_in_node_9 <= 0
 R300: - node_9_contains_any_point + x128_is_in_node_9 <= 0
 R301: - node_9_contains_any_point + x129_is_in_node_9 <= 0
 R302: - node_9_contains_any_point + x130_is_in_node_9 <= 0
 R303: - node_9_contains_any_point + x131_is_in_node_9 <= 0
 R304: - node_9_contains_any_point + x132_is_in_node_9 <= 0
 R305: - node_9_contains_any_point + x133_is_in_node_9 <= 0
 R306: - node_9_contains_any_point + x134_is_in_node_9 <= 0
 R307: - node_9_contains_any_point + x135_is_in_node_9 <= 0
 R308: - node_9_contains_any_point + x136_is_in_node_9 <= 0
 R309: - node_9_contains_any_point + x137_is_in_node_9 <= 0
 R310: - node_9_contains_any_point + x138_is_in_node_9 <= 0
 R311: - node_9_contains_any_point + x139_is_in_node_9 <= 0
 R312: - node_9_contains_any_point + x140_is_in_node_9 <= 0
 R313: - node_9_contains_any_point + x141_is_in_node_9 <= 0
 R314: - node_9_contains_any_point + x142_is_in_node_9 <= 0
 R315: - node_9_contains_any_point + x143_is_in_node_9 <= 0
 R316: - node_9_contains_any_point + x144_is_in_node_9 <= 0
 R317: - node_9_contains_any_point + x145_is_in_node_9 <= 0
 R318: - node_9_contains_any_point + x146_is_in_node_9 <= 0
 R319: - node_9_contains_any_point + x147_is_in_node_9 <= 0
 R320: - node_9_contains_any_point + x148_is_in_node_9 <= 0
 R321: - node_9_contains_any_point + x149_is_in_node_9 <= 0
 R322: - 7 node_9_contains_any_point + x0_is_in_node_9 + x1_is_in_node_9
   + x2_is_in_node_9 + x3_is_in_node_9 + x4_is_in_node_9 + x5_is_in_node_9
   + x6_is_in_node_9 + x7_is_in_node_9 + x8_is_in_node_9 + x9_is_in_node_9
   + x10_is_in_node_9 + x11_is_in_node_9 + x12_is_in_node_9
   + x13_is_in_node_9 + x14_is_in_node_9 + x15_is_in_node_9
   + x16_is_in_node_9 + x17_is_in_node_9 + x18_is_in_node_9
   + x19_is_in_node_9 + x20_is_in_node_9 + x21_is_in_node_9
   + x22_is_in_node_9 + x23_is_in_node_9 + x24_is_in_node_9
   + x25_is_in_node_9 + x26_is_in_node_9 + x27_is_in_node_9
   + x28_is_in_node_9 + x29_is_in_node_9 + x30_is_in_node_9
   + x31_is_in_node_9 + x32_is_in_node_9 + x33_is_in_node_9
   + x34_is_in_node_9 + x35_is_in_node_9 + x36_is_in_node_9
   + x37_is_in_node_9 + x38_is_in_node_9 + x39_is_in_node_9
   + x40_is_in_node_9 + x41_is_in_node_9 + x42_is_in_node_9
   + x43_is_in_node_9 + x44_is_in_node_9 + x45_is_in_node_9
   + x46_is_in_node_9 + x47_is_in_node_9 + x48_is_in_node_9
   + x49_is_in_node_9 + x50_is_in_node_9 + x51_is_in_node_9
   + x52_is_in_node_9 + x53_is_in_node_9 + x54_is_in_node_9
   + x55_is_in_node_9 + x56_is_in_node_9 + x57_is_in_node_9
   + x58_is_in_node_9 + x59_is_in_node_9 + x60_is_in_node_9
   + x61_is_in_node_9 + x62_is_in_node_9 + x63_is_in_node_9
   + x64_is_in_node_9 + x65_is_in_node_9 + x66_is_in_node_9
   + x67_is_in_node_9 + x68_is_in_node_9 + x69_is_in_node_9
   + x70_is_in_node_9 + x71_is_in_node_9 + x72_is_in_node_9
   + x73_is_in_node_9 + x74_is_in_node_9 + x75_is_in_node_9
   + x76_is_in_node_9 + x77_is_in_node_9 + x78_is_in_node_9
   + x79_is_in_node_9 + x80_is_in_node_9 + x81_is_in_node_9
   + x82_is_in_node_9 + x83_is_in_node_9 + x84_is_in_node_9
   + x85_is_in_node_9 + x86_is_in_node_9 + x87_is_in_node_9
   + x88_is_in_node_9 + x89_is_in_node_9 + x90_is_in_node_9
   + x91_is_in_node_9 + x92_is_in_node_9 + x93_is_in_node_9
   + x94_is_in_node_9 + x95_is_in_node_9 + x96_is_in_node_9
   + x97_is_in_node_9 + x98_is_in_node_9 + x99_is_in_node_9
   + x100_is_in_node_9 + x101_is_in_node_9 + x102_is_in_node_9
   + x103_is_in_node_9 + x104_is_in_node_9 + x105_is_in_node_9
   + x106_is_in_node_9 + x107_is_in_node_9 + x108_is_in_node_9
   + x109_is_in_node_9 + x110_is_in_node_9 + x111_is_in_node_9
   + x112_is_in_node_9 + x113_is_in_node_9 + x114_is_in_node_9
   + x115_is_in_node_9 + x116_is_in_node_9 + x117_is_in_node_9
   + x118_is_in_node_9 + x119_is_in_node_9 + x120_is_in_node_9
   + x121_is_in_node_9 + x122_is_in_node_9 + x123_is_in_node_9
   + x124_is_in_node_9 + x125_is_in_node_9 + x126_is_in_node_9
   + x127_is_in_node_9 + x128_is_in_node_9 + x129_is_in_node_9
   + x130_is_in_node_9 + x131_is_in_node_9 + x132_is_in_node_9
   + x133_is_in_node_9 + x134_is_in_node_9 + x135_is_in_node_9
   + x136_is_in_node_9 + x137_is_in_node_9 + x138_is_in_node_9
   + x139_is_in_node_9 + x140_is_in_node_9 + x141_is_in_node_9
   + x142_is_in_node_9 + x143_is_in_node_9 + x144_is_in_node_9
   + x145_is_in_node_9 + x146_is_in_node_9 + x147_is_in_node_9
   + x148_is_in_node_9 + x149_is_in_node_9 >= 0
 R323: x0_is_in_node_9 + x1_is_in_node_9 + x2_is_in_node_9
   + x3_is_in_node_9 + x4_is_in_node_9 + x5_is_in_node_9 + x6_is_in_node_9
   + x7_is_in_node_9 + x8_is_in_node_9 + x9_is_in_node_9 + x10_is_in_node_9
   + x11_is_in_node_9 + x12_is_in_node_9 + x13_is_in_node_9
   + x14_is_in_node_9 + x15_is_in_node_9 + x16_is_in_node_9
   + x17_is_in_node_9 + x18_is_in_node_9 + x19_is_in_node_9
   + x20_is_in_node_9 + x21_is_in_node_9 + x22_is_in_node_9
   + x23_is_in_node_9 + x24_is_in_node_9 + x25_is_in_node_9
   + x26_is_in_node_9 + x27_is_in_node_9 + x28_is_in_node_9
   + x29_is_in_node_9 + x30_is_in_node_9 + x31_is_in_node_9
   + x32_is_in_node_9 + x33_is_in_node_9 + x34_is_in_node_9
   + x35_is_in_node_9 + x36_is_in_node_9 + x37_is_in_node_9
   + x38_is_in_node_9 + x39_is_in_node_9 + x40_is_in_node_9
   + x41_is_in_node_9 + x42_is_in_node_9 + x43_is_in_node_9
   + x44_is_in_node_9 + x45_is_in_node_9 + x46_is_in_node_9
   + x47_is_in_node_9 + x48_is_in_node_9 + x49_is_in_node_9
   + x50_is_in_node_9 + x51_is_in_node_9 + x52_is_in_node_9
   + x53_is_in_node_9 + x54_is_in_node_9 + x55_is_in_node_9
   + x56_is_in_node_9 + x57_is_in_node_9 + x58_is_in_node_9
   + x59_is_in_node_9 + x60_is_in_node_9 + x61_is_in_node_9
   + x62_is_in_node_9 + x63_is_in_node_9 + x64_is_in_node_9
   + x65_is_in_node_9 + x66_is_in_node_9 + x67_is_in_node_9
   + x68_is_in_node_9 + x69_is_in_node_9 + x70_is_in_node_9
   + x71_is_in_node_9 + x72_is_in_node_9 + x73_is_in_node_9
   + x74_is_in_node_9 + x75_is_in_node_9 + x76_is_in_node_9
   + x77_is_in_node_9 + x78_is_in_node_9 + x79_is_in_node_9
   + x80_is_in_node_9 + x81_is_in_node_9 + x82_is_in_node_9
   + x83_is_in_node_9 + x84_is_in_node_9 + x85_is_in_node_9
   + x86_is_in_node_9 + x87_is_in_node_9 + x88_is_in_node_9
   + x89_is_in_node_9 + x90_is_in_node_9 + x91_is_in_node_9
   + x92_is_in_node_9 + x93_is_in_node_9 + x94_is_in_node_9
   + x95_is_in_node_9 + x96_is_in_node_9 + x97_is_in_node_9
   + x98_is_in_node_9 + x99_is_in_node_9 + x100_is_in_node_9
   + x101_is_in_node_9 + x102_is_in_node_9 + x103_is_in_node_9
   + x104_is_in_node_9 + x105_is_in_node_9 + x106_is_in_node_9
   + x107_is_in_node_9 + x108_is_in_node_9 + x109_is_in_node_9
   + x110_is_in_node_9 + x111_is_in_node_9 + x112_is_in_node_9
   + x113_is_in_node_9 + x114_is_in_node_9 + x115_is_in_node_9
   + x116_is_in_node_9 + x117_is_in_node_9 + x118_is_in_node_9
   + x119_is_in_node_9 + x120_is_in_node_9 + x121_is_in_node_9
   + x122_is_in_node_9 + x123_is_in_node_9 + x124_is_in_node_9
   + x125_is_in_node_9 + x126_is_in_node_9 + x127_is_in_node_9
   + x128_is_in_node_9 + x129_is_in_node_9 + x130_is_in_node_9
   + x131_is_in_node_9 + x132_is_in_node_9 + x133_is_in_node_9
   + x134_is_in_node_9 + x135_is_in_node_9 + x136_is_in_node_9
   + x137_is_in_node_9 + x138_is_in_node_9 + x139_is_in_node_9
   + x140_is_in_node_9 + x141_is_in_node_9 + x142_is_in_node_9
   + x143_is_in_node_9 + x144_is_in_node_9 + x145_is_in_node_9
   + x146_is_in_node_9 + x147_is_in_node_9 + x148_is_in_node_9
   + x149_is_in_node_9 = 1
 R324: - node_10_contains_any_point + x0_is_in_node_10 <= 0
 R325: - node_10_contains_any_point + x1_is_in_node_10 <= 0
 R326: - node_10_contains_any_point + x2_is_in_node_10 <= 0
 R327: - node_10_contains_any_point + x3_is_in_node_10 <= 0
 R328: - node_10_contains_any_point + x4_is_in_node_10 <= 0
 R329: - node_10_contains_any_point + x5_is_in_node_10 <= 0
 R330: - node_10_contains_any_point + x6_is_in_node_10 <= 0
 R331: - node_10_contains_any_point + x7_is_in_node_10 <= 0
 R332: - node_10_contains_any_point + x8_is_in_node_10 <= 0
 R333: - node_10_contains_any_point + x9_is_in_node_10 <= 0
 R334: - node_10_contains_any_point + x10_is_in_node_10 <= 0
 R335: - node_10_contains_any_point + x11_is_in_node_10 <= 0
 R336: - node_10_contains_any_point + x12_is_in_node_10 <= 0
 R337: - node_10_contains_any_point + x13_is_in_node_10 <= 0
 R338: - node_10_contains_any_point + x14_is_in_node_10 <= 0
 R339: - node_10_contains_any_point + x15_is_in_node_10 <= 0
 R340: - node_10_contains_any_point + x16_is_in_node_10 <= 0
 R341: - node_10_contains_any_point + x17_is_in_node_10 <= 0
 R342: - node_10_contains_any_point + x18_is_in_node_10 <= 0
 R343: - node_10_contains_any_point + x19_is_in_node_10 <= 0
 R344: - node_10_contains_any_point + x20_is_in_node_10 <= 0
 R345: - node_10_contains_any_point + x21_is_in_node_10 <= 0
 R346: - node_10_contains_any_point + x22_is_in_node_10 <= 0
 R347: - node_10_contains_any_point + x23_is_in_node_10 <= 0
 R348: - node_10_contains_any_point + x24_is_in_node_10 <= 0
 R349: - node_10_contains_any_point + x25_is_in_node_10 <= 0
 R350: - node_10_contains_any_point + x26_is_in_node_10 <= 0
 R351: - node_10_contains_any_point + x27_is_in_node_10 <= 0
 R352: - node_10_contains_any_point + x28_is_in_node_10 <= 0
 R353: - node_10_contains_any_point + x29_is_in_node_10 <= 0
 R354: - node_10_contains_any_point + x30_is_in_node_10 <= 0
 R355: - node_10_contains_any_point + x31_is_in_node_10 <= 0
 R356: - node_10_contains_any_point + x32_is_in_node_10 <= 0
 R357: - node_10_contains_any_point + x33_is_in_node_10 <= 0
 R358: - node_10_contains_any_point + x34_is_in_node_10 <= 0
 R359: - node_10_contains_any_point + x35_is_in_node_10 <= 0
 R360: - node_10_contains_any_point + x36_is_in_node_10 <= 0
 R361: - node_10_contains_any_point + x37_is_in_node_10 <= 0
 R362: - node_10_contains_any_point + x38_is_in_node_10 <= 0
 R363: - node_10_contains_any_point + x39_is_in_node_10 <= 0
 R364: - node_10_contains_any_point + x40_is_in_node_10 <= 0
 R365: - node_10_contains_any_point + x41_is_in_node_10 <= 0
 R366: - node_10_contains_any_point + x42_is_in_node_10 <= 0
 R367: - node_10_contains_any_point + x43_is_in_node_10 <= 0
 R368: - node_10_contains_any_point + x44_is_in_node_10 <= 0
 R369: - node_10_contains_any_point + x45_is_in_node_10 <= 0
 R370: - node_10_contains_any_point + x46_is_in_node_10 <= 0
 R371: - node_10_contains_any_point + x47_is_in_node_10 <= 0
 R372: - node_10_contains_any_point + x48_is_in_node_10 <= 0
 R373: - node_10_contains_any_point + x49_is_in_node_10 <= 0
 R374: - node_10_contains_any_point + x50_is_in_node_10 <= 0
 R375: - node_10_contains_any_point + x51_is_in_node_10 <= 0
 R376: - node_10_contains_any_point + x52_is_in_node_10 <= 0
 R377: - node_10_contains_any_point + x53_is_in_node_10 <= 0
 R378: - node_10_contains_any_point + x54_is_in_node_10 <= 0
 R379: - node_10_contains_any_point + x55_is_in_node_10 <= 0
 R380: - node_10_contains_any_point + x56_is_in_node_10 <= 0
 R381: - node_10_contains_any_point + x57_is_in_node_10 <= 0
 R382: - node_10_contains_any_point + x58_is_in_node_10 <= 0
 R383: - node_10_contains_any_point + x59_is_in_node_10 <= 0
 R384: - node_10_contains_any_point + x60_is_in_node_10 <= 0
 R385: - node_10_contains_any_point + x61_is_in_node_10 <= 0
 R386: - node_10_contains_any_point + x62_is_in_node_10 <= 0
 R387: - node_10_contains_any_point + x63_is_in_node_10 <= 0
 R388: - node_10_contains_any_point + x64_is_in_node_10 <= 0
 R389: - node_10_contains_any_point + x65_is_in_node_10 <= 0
 R390: - node_10_contains_any_point + x66_is_in_node_10 <= 0
 R391: - node_10_contains_any_point + x67_is_in_node_10 <= 0
 R392: - node_10_contains_any_point + x68_is_in_node_10 <= 0
 R393: - node_10_contains_any_point + x69_is_in_node_10 <= 0
 R394: - node_10_contains_any_point + x70_is_in_node_10 <= 0
 R395: - node_10_contains_any_point + x71_is_in_node_10 <= 0
 R396: - node_10_contains_any_point + x72_is_in_node_10 <= 0
 R397: - node_10_contains_any_point + x73_is_in_node_10 <= 0
 R398: - node_10_contains_any_point + x74_is_in_node_10 <= 0
 R399: - node_10_contains_any_point + x75_is_in_node_10 <= 0
 R400: - node_10_contains_any_point + x76_is_in_node_10 <= 0
 R401: - node_10_contains_any_point + x77_is_in_node_10 <= 0
 R402: - node_10_contains_any_point + x78_is_in_node_10 <= 0
 R403: - node_10_contains_any_point + x79_is_in_node_10 <= 0
 R404: - node_10_contains_any_point + x80_is_in_node_10 <= 0
 R405: - node_10_contains_any_point + x81_is_in_node_10 <= 0
 R406: - node_10_contains_any_point + x82_is_in_node_10 <= 0
 R407: - node_10_contains_any_point + x83_is_in_node_10 <= 0
 R408: - node_10_contains_any_point + x84_is_in_node_10 <= 0
 R409: - node_10_contains_any_point + x85_is_in_node_10 <= 0
 R410: - node_10_contains_any_point + x86_is_in_node_10 <= 0
 R411: - node_10_contains_any_point + x87_is_in_node_10 <= 0
 R412: - node_10_contains_any_point + x88_is_in_node_10 <= 0
 R413: - node_10_contains_any_point + x89_is_in_node_10 <= 0
 R414: - node_10_contains_any_point + x90_is_in_node_10 <= 0
 R415: - node_10_contains_any_point + x91_is_in_node_10 <= 0
 R416: - node_10_contains_any_point + x92_is_in_node_10 <= 0
 R417: - node_10_contains_any_point + x93_is_in_node_10 <= 0
 R418: - node_10_contains_any_point + x94_is_in_node_10 <= 0
 R419: - node_10_contains_any_point + x95_is_in_node_10 <= 0
 R420: - node_10_contains_any_point + x96_is_in_node_10 <= 0
 R421: - node_10_contains_any_point + x97_is_in_node_10 <= 0
 R422: - node_10_contains_any_point + x98_is_in_node_10 <= 0
 R423: - node_10_contains_any_point + x99_is_in_node_10 <= 0
 R424: - node_10_contains_any_point + x100_is_in_node_10 <= 0
 R425: - node_10_contains_any_point + x101_is_in_node_10 <= 0
 R426: - node_10_contains_any_point + x102_is_in_node_10 <= 0
 R427: - node_10_contains_any_point + x103_is_in_node_10 <= 0
 R428: - node_10_contains_any_point + x104_is_in_node_10 <= 0
 R429: - node_10_contains_any_point + x105_is_in_node_10 <= 0
 R430: - node_10_contains_any_point + x106_is_in_node_10 <= 0
 R431: - node_10_contains_any_point + x107_is_in_node_10 <= 0
 R432: - node_10_contains_any_point + x108_is_in_node_10 <= 0
 R433: - node_10_contains_any_point + x109_is_in_node_10 <= 0
 R434: - node_10_contains_any_point + x110_is_in_node_10 <= 0
 R435: - node_10_contains_any_point + x111_is_in_node_10 <= 0
 R436: - node_10_contains_any_point + x112_is_in_node_10 <= 0
 R437: - node_10_contains_any_point + x113_is_in_node_10 <= 0
 R438: - node_10_contains_any_point + x114_is_in_node_10 <= 0
 R439: - node_10_contains_any_point + x115_is_in_node_10 <= 0
 R440: - node_10_contains_any_point + x116_is_in_node_10 <= 0
 R441: - node_10_contains_any_point + x117_is_in_node_10 <= 0
 R442: - node_10_contains_any_point + x118_is_in_node_10 <= 0
 R443: - node_10_contains_any_point + x119_is_in_node_10 <= 0
 R444: - node_10_contains_any_point + x120_is_in_node_10 <= 0
 R445: - node_10_contains_any_point + x121_is_in_node_10 <= 0
 R446: - node_10_contains_any_point + x122_is_in_node_10 <= 0
 R447: - node_10_contains_any_point + x123_is_in_node_10 <= 0
 R448: - node_10_contains_any_point + x124_is_in_node_10 <= 0
 R449: - node_10_contains_any_point + x125_is_in_node_10 <= 0
 R450: - node_10_contains_any_point + x126_is_in_node_10 <= 0
 R451: - node_10_contains_any_point + x127_is_in_node_10 <= 0
 R452: - node_10_contains_any_point + x128_is_in_node_10 <= 0
 R453: - node_10_contains_any_point + x129_is_in_node_10 <= 0
 R454: - node_10_contains_any_point + x130_is_in_node_10 <= 0
 R455: - node_10_contains_any_point + x131_is_in_node_10 <= 0
 R456: - node_10_contains_any_point + x132_is_in_node_10 <= 0
 R457: - node_10_contains_any_point + x133_is_in_node_10 <= 0
 R458: - node_10_contains_any_point + x134_is_in_node_10 <= 0
 R459: - node_10_contains_any_point + x135_is_in_node_10 <= 0
 R460: - node_10_contains_any_point + x136_is_in_node_10 <= 0
 R461: - node_10_contains_any_point + x137_is_in_node_10 <= 0
 R462: - node_10_contains_any_point + x138_is_in_node_10 <= 0
 R463: - node_10_contains_any_point + x139_is_in_node_10 <= 0
 R464: - node_10_contains_any_point + x140_is_in_node_10 <= 0
 R465: - node_10_contains_any_point + x141_is_in_node_10 <= 0
 R466: - node_10_contains_any_point + x142_is_in_node_10 <= 0
 R467: - node_10_contains_any_point + x143_is_in_node_10 <= 0
 R468: - node_10_contains_any_point + x144_is_in_node_10 <= 0
 R469: - node_10_contains_any_point + x145_is_in_node_10 <= 0
 R470: - node_10_contains_any_point + x146_is_in_node_10 <= 0
 R471: - node_10_contains_any_point + x147_is_in_node_10 <= 0
 R472: - node_10_contains_any_point + x148_is_in_node_10 <= 0
 R473: - node_10_contains_any_point + x149_is_in_node_10 <= 0
 R474: - 7 node_10_contains_any_point + x0_is_in_node_10 + x1_is_in_node_10
   + x2_is_in_node_10 + x3_is_in_node_10 + x4_is_in_node_10
   + x5_is_in_node_10 + x6_is_in_node_10 + x7_is_in_node_10
   + x8_is_in_node_10 + x9_is_in_node_10 + x10_is_in_node_10
   + x11_is_in_node_10 + x12_is_in_node_10 + x13_is_in_node_10
   + x14_is_in_node_10 + x15_is_in_node_10 + x16_is_in_node_10
   + x17_is_in_node_10 + x18_is_in_node_10 + x19_is_in_node_10
   + x20_is_in_node_10 + x21_is_in_node_10 + x22_is_in_node_10
   + x23_is_in_node_10 + x24_is_in_node_10 + x25_is_in_node_10
   + x26_is_in_node_10 + x27_is_in_node_10 + x28_is_in_node_10
   + x29_is_in_node_10 + x30_is_in_node_10 + x31_is_in_node_10
   + x32_is_in_node_10 + x33_is_in_node_10 + x34_is_in_node_10
   + x35_is_in_node_10 + x36_is_in_node_10 + x37_is_in_node_10
   + x38_is_in_node_10 + x39_is_in_node_10 + x40_is_in_node_10
   + x41_is_in_node_10 + x42_is_in_node_10 + x43_is_in_node_10
   + x44_is_in_node_10 + x45_is_in_node_10 + x46_is_in_node_10
   + x47_is_in_node_10 + x48_is_in_node_10 + x49_is_in_node_10
   + x50_is_in_node_10 + x51_is_in_node_10 + x52_is_in_node_10
   + x53_is_in_node_10 + x54_is_in_node_10 + x55_is_in_node_10
   + x56_is_in_node_10 + x57_is_in_node_10 + x58_is_in_node_10
   + x59_is_in_node_10 + x60_is_in_node_10 + x61_is_in_node_10
   + x62_is_in_node_10 + x63_is_in_node_10 + x64_is_in_node_10
   + x65_is_in_node_10 + x66_is_in_node_10 + x67_is_in_node_10
   + x68_is_in_node_10 + x69_is_in_node_10 + x70_is_in_node_10
   + x71_is_in_node_10 + x72_is_in_node_10 + x73_is_in_node_10
   + x74_is_in_node_10 + x75_is_in_node_10 + x76_is_in_node_10
   + x77_is_in_node_10 + x78_is_in_node_10 + x79_is_in_node_10
   + x80_is_in_node_10 + x81_is_in_node_10 + x82_is_in_node_10
   + x83_is_in_node_10 + x84_is_in_node_10 + x85_is_in_node_10
   + x86_is_in_node_10 + x87_is_in_node_10 + x88_is_in_node_10
   + x89_is_in_node_10 + x90_is_in_node_10 + x91_is_in_node_10
   + x92_is_in_node_10 + x93_is_in_node_10 + x94_is_in_node_10
   + x95_is_in_node_10 + x96_is_in_node_10 + x97_is_in_node_10
   + x98_is_in_node_10 + x99_is_in_node_10 + x100_is_in_node_10
   + x101_is_in_node_10 + x102_is_in_node_10 + x103_is_in_node_10
   + x104_is_in_node_10 + x105_is_in_node_10 + x106_is_in_node_10
   + x107_is_in_node_10 + x108_is_in_node_10 + x109_is_in_node_10
   + x110_is_in_node_10 + x111_is_in_node_10 + x112_is_in_node_10
   + x113_is_in_node_10 + x114_is_in_node_10 + x115_is_in_node_10
   + x116_is_in_node_10 + x117_is_in_node_10 + x118_is_in_node_10
   + x119_is_in_node_10 + x120_is_in_node_10 + x121_is_in_node_10
   + x122_is_in_node_10 + x123_is_in_node_10 + x124_is_in_node_10
   + x125_is_in_node_10 + x126_is_in_node_10 + x127_is_in_node_10
   + x128_is_in_node_10 + x129_is_in_node_10 + x130_is_in_node_10
   + x131_is_in_node_10 + x132_is_in_node_10 + x133_is_in_node_10
   + x134_is_in_node_10 + x135_is_in_node_10 + x136_is_in_node_10
   + x137_is_in_node_10 + x138_is_in_node_10 + x139_is_in_node_10
   + x140_is_in_node_10 + x141_is_in_node_10 + x142_is_in_node_10
   + x143_is_in_node_10 + x144_is_in_node_10 + x145_is_in_node_10
   + x146_is_in_node_10 + x147_is_in_node_10 + x148_is_in_node_10
   + x149_is_in_node_10 >= 0
 R475: x0_is_in_node_10 + x1_is_in_node_10 + x2_is_in_node_10
   + x3_is_in_node_10 + x4_is_in_node_10 + x5_is_in_node_10
   + x6_is_in_node_10 + x7_is_in_node_10 + x8_is_in_node_10
   + x9_is_in_node_10 + x10_is_in_node_10 + x11_is_in_node_10
   + x12_is_in_node_10 + x13_is_in_node_10 + x14_is_in_node_10
   + x15_is_in_node_10 + x16_is_in_node_10 + x17_is_in_node_10
   + x18_is_in_node_10 + x19_is_in_node_10 + x20_is_in_node_10
   + x21_is_in_node_10 + x22_is_in_node_10 + x23_is_in_node_10
   + x24_is_in_node_10 + x25_is_in_node_10 + x26_is_in_node_10
   + x27_is_in_node_10 + x28_is_in_node_10 + x29_is_in_node_10
   + x30_is_in_node_10 + x31_is_in_node_10 + x32_is_in_node_10
   + x33_is_in_node_10 + x34_is_in_node_10 + x35_is_in_node_10
   + x36_is_in_node_10 + x37_is_in_node_10 + x38_is_in_node_10
   + x39_is_in_node_10 + x40_is_in_node_10 + x41_is_in_node_10
   + x42_is_in_node_10 + x43_is_in_node_10 + x44_is_in_node_10
   + x45_is_in_node_10 + x46_is_in_node_10 + x47_is_in_node_10
   + x48_is_in_node_10 + x49_is_in_node_10 + x50_is_in_node_10
   + x51_is_in_node_10 + x52_is_in_node_10 + x53_is_in_node_10
   + x54_is_in_node_10 + x55_is_in_node_10 + x56_is_in_node_10
   + x57_is_in_node_10 + x58_is_in_node_10 + x59_is_in_node_10
   + x60_is_in_node_10 + x61_is_in_node_10 + x62_is_in_node_10
   + x63_is_in_node_10 + x64_is_in_node_10 + x65_is_in_node_10
   + x66_is_in_node_10 + x67_is_in_node_10 + x68_is_in_node_10
   + x69_is_in_node_10 + x70_is_in_node_10 + x71_is_in_node_10
   + x72_is_in_node_10 + x73_is_in_node_10 + x74_is_in_node_10
   + x75_is_in_node_10 + x76_is_in_node_10 + x77_is_in_node_10
   + x78_is_in_node_10 + x79_is_in_node_10 + x80_is_in_node_10
   + x81_is_in_node_10 + x82_is_in_node_10 + x83_is_in_node_10
   + x84_is_in_node_10 + x85_is_in_node_10 + x86_is_in_node_10
   + x87_is_in_node_10 + x88_is_in_node_10 + x89_is_in_node_10
   + x90_is_in_node_10 + x91_is_in_node_10 + x92_is_in_node_10
   + x93_is_in_node_10 + x94_is_in_node_10 + x95_is_in_node_10
   + x96_is_in_node_10 + x97_is_in_node_10 + x98_is_in_node_10
   + x99_is_in_node_10 + x100_is_in_node_10 + x101_is_in_node_10
   + x102_is_in_node_10 + x103_is_in_node_10 + x104_is_in_node_10
   + x105_is_in_node_10 + x106_is_in_node_10 + x107_is_in_node_10
   + x108_is_in_node_10 + x109_is_in_node_10 + x110_is_in_node_10
   + x111_is_in_node_10 + x112_is_in_node_10 + x113_is_in_node_10
   + x114_is_in_node_10 + x115_is_in_node_10 + x116_is_in_node_10
   + x117_is_in_node_10 + x118_is_in_node_10 + x119_is_in_node_10
   + x120_is_in_node_10 + x121_is_in_node_10 + x122_is_in_node_10
   + x123_is_in_node_10 + x124_is_in_node_10 + x125_is_in_node_10
   + x126_is_in_node_10 + x127_is_in_node_10 + x128_is_in_node_10
   + x129_is_in_node_10 + x130_is_in_node_10 + x131_is_in_node_10
   + x132_is_in_node_10 + x133_is_in_node_10 + x134_is_in_node_10
   + x135_is_in_node_10 + x136_is_in_node_10 + x137_is_in_node_10
   + x138_is_in_node_10 + x139_is_in_node_10 + x140_is_in_node_10
   + x141_is_in_node_10 + x142_is_in_node_10 + x143_is_in_node_10
   + x144_is_in_node_10 + x145_is_in_node_10 + x146_is_in_node_10
   + x147_is_in_node_10 + x148_is_in_node_10 + x149_is_in_node_10 = 1
 R476: - node_11_contains_any_point + x0_is_in_node_11 <= 0
 R477: - node_11_contains_any_point + x1_is_in_node_11 <= 0
 R478: - node_11_contains_any_point + x2_is_in_node_11 <= 0
 R479: - node_11_contains_any_point + x3_is_in_node_11 <= 0
 R480: - node_11_contains_any_point + x4_is_in_node_11 <= 0
 R481: - node_11_contains_any_point + x5_is_in_node_11 <= 0
 R482: - node_11_contains_any_point + x6_is_in_node_11 <= 0
 R483: - node_11_contains_any_point + x7_is_in_node_11 <= 0
 R484: - node_11_contains_any_point + x8_is_in_node_11 <= 0
 R485: - node_11_contains_any_point + x9_is_in_node_11 <= 0
 R486: - node_11_contains_any_point + x10_is_in_node_11 <= 0
 R487: - node_11_contains_any_point + x11_is_in_node_11 <= 0
 R488: - node_11_contains_any_point + x12_is_in_node_11 <= 0
 R489: - node_11_contains_any_point + x13_is_in_node_11 <= 0
 R490: - node_11_contains_any_point + x14_is_in_node_11 <= 0
 R491: - node_11_contains_any_point + x15_is_in_node_11 <= 0
 R492: - node_11_contains_any_point + x16_is_in_node_11 <= 0
 R493: - node_11_contains_any_point + x17_is_in_node_11 <= 0
 R494: - node_11_contains_any_point + x18_is_in_node_11 <= 0
 R495: - node_11_contains_any_point + x19_is_in_node_11 <= 0
 R496: - node_11_contains_any_point + x20_is_in_node_11 <= 0
 R497: - node_11_contains_any_point + x21_is_in_node_11 <= 0
 R498: - node_11_contains_any_point + x22_is_in_node_11 <= 0
 R499: - node_11_contains_any_point + x23_is_in_node_11 <= 0
 R500: - node_11_contains_any_point + x24_is_in_node_11 <= 0
 R501: - node_11_contains_any_point + x25_is_in_node_11 <= 0
 R502: - node_11_contains_any_point + x26_is_in_node_11 <= 0
 R503: - node_11_contains_any_point + x27_is_in_node_11 <= 0
 R504: - node_11_contains_any_point + x28_is_in_node_11 <= 0
 R505: - node_11_contains_any_point + x29_is_in_node_11 <= 0
 R506: - node_11_contains_any_point + x30_is_in_node_11 <= 0
 R507: - node_11_contains_any_point + x31_is_in_node_11 <= 0
 R508: - node_11_contains_any_point + x32_is_in_node_11 <= 0
 R509: - node_11_contains_any_point + x33_is_in_node_11 <= 0
 R510: - node_11_contains_any_point + x34_is_in_node_11 <= 0
 R511: - node_11_contains_any_point + x35_is_in_node_11 <= 0
 R512: - node_11_contains_any_point + x36_is_in_node_11 <= 0
 R513: - node_11_contains_any_point + x37_is_in_node_11 <= 0
 R514: - node_11_contains_any_point + x38_is_in_node_11 <= 0
 R515: - node_11_contains_any_point + x39_is_in_node_11 <= 0
 R516: - node_11_contains_any_point + x40_is_in_node_11 <= 0
 R517: - node_11_contains_any_point + x41_is_in_node_11 <= 0
 R518: - node_11_contains_any_point + x42_is_in_node_11 <= 0
 R519: - node_11_contains_any_point + x43_is_in_node_11 <= 0
 R520: - node_11_contains_any_point + x44_is_in_node_11 <= 0
 R521: - node_11_contains_any_point + x45_is_in_node_11 <= 0
 R522: - node_11_contains_any_point + x46_is_in_node_11 <= 0
 R523: - node_11_contains_any_point + x47_is_in_node_11 <= 0
 R524: - node_11_contains_any_point + x48_is_in_node_11 <= 0
 R525: - node_11_contains_any_point + x49_is_in_node_11 <= 0
 R526: - node_11_contains_any_point + x50_is_in_node_11 <= 0
 R527: - node_11_contains_any_point + x51_is_in_node_11 <= 0
 R528: - node_11_contains_any_point + x52_is_in_node_11 <= 0
 R529: - node_11_contains_any_point + x53_is_in_node_11 <= 0
 R530: - node_11_contains_any_point + x54_is_in_node_11 <= 0
 R531: - node_11_contains_any_point + x55_is_in_node_11 <= 0
 R532: - node_11_contains_any_point + x56_is_in_node_11 <= 0
 R533: - node_11_contains_any_point + x57_is_in_node_11 <= 0
 R534: - node_11_contains_any_point + x58_is_in_node_11 <= 0
 R535: - node_11_contains_any_point + x59_is_in_node_11 <= 0
 R536: - node_11_contains_any_point + x60_is_in_node_11 <= 0
 R537: - node_11_contains_any_point + x61_is_in_node_11 <= 0
 R538: - node_11_contains_any_point + x62_is_in_node_11 <= 0
 R539: - node_11_contains_any_point + x63_is_in_node_11 <= 0
 R540: - node_11_contains_any_point + x64_is_in_node_11 <= 0
 R541: - node_11_contains_any_point + x65_is_in_node_11 <= 0
 R542: - node_11_contains_any_point + x66_is_in_node_11 <= 0
 R543: - node_11_contains_any_point + x67_is_in_node_11 <= 0
 R544: - node_11_contains_any_point + x68_is_in_node_11 <= 0
 R545: - node_11_contains_any_point + x69_is_in_node_11 <= 0
 R546: - node_11_contains_any_point + x70_is_in_node_11 <= 0
 R547: - node_11_contains_any_point + x71_is_in_node_11 <= 0
 R548: - node_11_contains_any_point + x72_is_in_node_11 <= 0
 R549: - node_11_contains_any_point + x73_is_in_node_11 <= 0
 R550: - node_11_contains_any_point + x74_is_in_node_11 <= 0
 R551: - node_11_contains_any_point + x75_is_in_node_11 <= 0
 R552: - node_11_contains_any_point + x76_is_in_node_11 <= 0
 R553: - node_11_contains_any_point + x77_is_in_node_11 <= 0
 R554: - node_11_contains_any_point + x78_is_in_node_11 <= 0
 R555: - node_11_contains_any_point + x79_is_in_node_11 <= 0
 R556: - node_11_contains_any_point + x80_is_in_node_11 <= 0
 R557: - node_11_contains_any_point + x81_is_in_node_11 <= 0
 R558: - node_11_contains_any_point + x82_is_in_node_11 <= 0
 R559: - node_11_contains_any_point + x83_is_in_node_11 <= 0
 R560: - node_11_contains_any_point + x84_is_in_node_11 <= 0
 R561: - node_11_contains_any_point + x85_is_in_node_11 <= 0
 R562: - node_11_contains_any_point + x86_is_in_node_11 <= 0
 R563: - node_11_contains_any_point + x87_is_in_node_11 <= 0
 R564: - node_11_contains_any_point + x88_is_in_node_11 <= 0
 R565: - node_11_contains_any_point + x89_is_in_node_11 <= 0
 R566: - node_11_contains_any_point + x90_is_in_node_11 <= 0
 R567: - node_11_contains_any_point + x91_is_in_node_11 <= 0
 R568: - node_11_contains_any_point + x92_is_in_node_11 <= 0
 R569: - node_11_contains_any_point + x93_is_in_node_11 <= 0
 R570: - node_11_contains_any_point + x94_is_in_node_11 <= 0
 R571: - node_11_contains_any_point + x95_is_in_node_11 <= 0
 R572: - node_11_contains_any_point + x96_is_in_node_11 <= 0
 R573: - node_11_contains_any_point + x97_is_in_node_11 <= 0
 R574: - node_11_contains_any_point + x98_is_in_node_11 <= 0
 R575: - node_11_contains_any_point + x99_is_in_node_11 <= 0
 R576: - node_11_contains_any_point + x100_is_in_node_11 <= 0
 R577: - node_11_contains_any_point + x101_is_in_node_11 <= 0
 R578: - node_11_contains_any_point + x102_is_in_node_11 <= 0
 R579: - node_11_contains_any_point + x103_is_in_node_11 <= 0
 R580: - node_11_contains_any_point + x104_is_in_node_11 <= 0
 R581: - node_11_contains_any_point + x105_is_in_node_11 <= 0
 R582: - node_11_contains_any_point + x106_is_in_node_11 <= 0
 R583: - node_11_contains_any_point + x107_is_in_node_11 <= 0
 R584: - node_11_contains_any_point + x108_is_in_node_11 <= 0
 R585: - node_11_contains_any_point + x109_is_in_node_11 <= 0
 R586: - node_11_contains_any_point + x110_is_in_node_11 <= 0
 R587: - node_11_contains_any_point + x111_is_in_node_11 <= 0
 R588: - node_11_contains_any_point + x112_is_in_node_11 <= 0
 R589: - node_11_contains_any_point + x113_is_in_node_11 <= 0
 R590: - node_11_contains_any_point + x114_is_in_node_11 <= 0
 R591: - node_11_contains_any_point + x115_is_in_node_11 <= 0
 R592: - node_11_contains_any_point + x116_is_in_node_11 <= 0
 R593: - node_11_contains_any_point + x117_is_in_node_11 <= 0
 R594: - node_11_contains_any_point + x118_is_in_node_11 <= 0
 R595: - node_11_contains_any_point + x119_is_in_node_11 <= 0
 R596: - node_11_contains_any_point + x120_is_in_node_11 <= 0
 R597: - node_11_contains_any_point + x121_is_in_node_11 <= 0
 R598: - node_11_contains_any_point + x122_is_in_node_11 <= 0
 R599: - node_11_contains_any_point + x123_is_in_node_11 <= 0
 R600: - node_11_contains_any_point + x124_is_in_node_11 <= 0
 R601: - node_11_contains_any_point + x125_is_in_node_11 <= 0
 R602: - node_11_contains_any_point + x126_is_in_node_11 <= 0
 R603: - node_11_contains_any_point + x127_is_in_node_11 <= 0
 R604: - node_11_contains_any_point + x128_is_in_node_11 <= 0
 R605: - node_11_contains_any_point + x129_is_in_node_11 <= 0
 R606: - node_11_contains_any_point + x130_is_in_node_11 <= 0
 R607: - node_11_contains_any_point + x131_is_in_node_11 <= 0
 R608: - node_11_contains_any_point + x132_is_in_node_11 <= 0
 R609: - node_11_contains_any_point + x133_is_in_node_11 <= 0
 R610: - node_11_contains_any_point + x134_is_in_node_11 <= 0
 R611: - node_11_contains_any_point + x135_is_in_node_11 <= 0
 R612: - node_11_contains_any_point + x136_is_in_node_11 <= 0
 R613: - node_11_contains_any_point + x137_is_in_node_11 <= 0
 R614: - node_11_contains_any_point + x138_is_in_node_11 <= 0
 R615: - node_11_contains_any_point + x139_is_in_node_11 <= 0
 R616: - node_11_contains_any_point + x140_is_in_node_11 <= 0
 R617: - node_11_contains_any_point + x141_is_in_node_11 <= 0
 R618: - node_11_contains_any_point + x142_is_in_node_11 <= 0
 R619: - node_11_contains_any_point + x143_is_in_node_11 <= 0
 R620: - node_11_contains_any_point + x144_is_in_node_11 <= 0
 R621: - node_11_contains_any_point + x145_is_in_node_11 <= 0
 R622: - node_11_contains_any_point + x146_is_in_node_11 <= 0
 R623: - node_11_contains_any_point + x147_is_in_node_11 <= 0
 R624: - node_11_contains_any_point + x148_is_in_node_11 <= 0
 R625: - node_11_contains_any_point + x149_is_in_node_11 <= 0
 R626: - 7 node_11_contains_any_point + x0_is_in_node_11 + x1_is_in_node_11
   + x2_is_in_node_11 + x3_is_in_node_11 + x4_is_in_node_11
   + x5_is_in_node_11 + x6_is_in_node_11 + x7_is_in_node_11
   + x8_is_in_node_11 + x9_is_in_node_11 + x10_is_in_node_11
   + x11_is_in_node_11 + x12_is_in_node_11 + x13_is_in_node_11
   + x14_is_in_node_11 + x15_is_in_node_11 + x16_is_in_node_11
   + x17_is_in_node_11 + x18_is_in_node_11 + x19_is_in_node_11
   + x20_is_in_node_11 + x21_is_in_node_11 + x22_is_in_node_11
   + x23_is_in_node_11 + x24_is_in_node_11 + x25_is_in_node_11
   + x26_is_in_node_11 + x27_is_in_node_11 + x28_is_in_node_11
   + x29_is_in_node_11 + x30_is_in_node_11 + x31_is_in_node_11
   + x32_is_in_node_11 + x33_is_in_node_11 + x34_is_in_node_11
   + x35_is_in_node_11 + x36_is_in_node_11 + x37_is_in_node_11
   + x38_is_in_node_11 + x39_is_in_node_11 + x40_is_in_node_11
   + x41_is_in_node_11 + x42_is_in_node_11 + x43_is_in_node_11
   + x44_is_in_node_11 + x45_is_in_node_11 + x46_is_in_node_11
   + x47_is_in_node_11 + x48_is_in_node_11 + x49_is_in_node_11
   + x50_is_in_node_11 + x51_is_in_node_11 + x52_is_in_node_11
   + x53_is_in_node_11 + x54_is_in_node_11 + x55_is_in_node_11
   + x56_is_in_node_11 + x57_is_in_node_11 + x58_is_in_node_11
   + x59_is_in_node_11 + x60_is_in_node_11 + x61_is_in_node_11
   + x62_is_in_node_11 + x63_is_in_node_11 + x64_is_in_node_11
   + x65_is_in_node_11 + x66_is_in_node_11 + x67_is_in_node_11
   + x68_is_in_node_11 + x69_is_in_node_11 + x70_is_in_node_11
   + x71_is_in_node_11 + x72_is_in_node_11 + x73_is_in_node_11
   + x74_is_in_node_11 + x75_is_in_node_11 + x76_is_in_node_11
   + x77_is_in_node_11 + x78_is_in_node_11 + x79_is_in_node_11
   + x80_is_in_node_11 + x81_is_in_node_11 + x82_is_in_node_11
   + x83_is_in_node_11 + x84_is_in_node_11 + x85_is_in_node_11
   + x86_is_in_node_11 + x87_is_in_node_11 + x88_is_in_node_11
   + x89_is_in_node_11 + x90_is_in_node_11 + x91_is_in_node_11
   + x92_is_in_node_11 + x93_is_in_node_11 + x94_is_in_node_11
   + x95_is_in_node_11 + x96_is_in_node_11 + x97_is_in_node_11
   + x98_is_in_node_11 + x99_is_in_node_11 + x100_is_in_node_11
   + x101_is_in_node_11 + x102_is_in_node_11 + x103_is_in_node_11
   + x104_is_in_node_11 + x105_is_in_node_11 + x106_is_in_node_11
   + x107_is_in_node_11 + x108_is_in_node_11 + x109_is_in_node_11
   + x110_is_in_node_11 + x111_is_in_node_11 + x112_is_in_node_11
   + x113_is_in_node_11 + x114_is_in_node_11 + x115_is_in_node_11
   + x116_is_in_node_11 + x117_is_in_node_11 + x118_is_in_node_11
   + x119_is_in_node_11 + x120_is_in_node_11 + x121_is_in_node_11
   + x122_is_in_node_11 + x123_is_in_node_11 + x124_is_in_node_11
   + x125_is_in_node_11 + x126_is_in_node_11 + x127_is_in_node_11
   + x128_is_in_node_11 + x129_is_in_node_11 + x130_is_in_node_11
   + x131_is_in_node_11 + x132_is_in_node_11 + x133_is_in_node_11
   + x134_is_in_node_11 + x135_is_in_node_11 + x136_is_in_node_11
   + x137_is_in_node_11 + x138_is_in_node_11 + x139_is_in_node_11
   + x140_is_in_node_11 + x141_is_in_node_11 + x142_is_in_node_11
   + x143_is_in_node_11 + x144_is_in_node_11 + x145_is_in_node_11
   + x146_is_in_node_11 + x147_is_in_node_11 + x148_is_in_node_11
   + x149_is_in_node_11 >= 0
 R627: x0_is_in_node_11 + x1_is_in_node_11 + x2_is_in_node_11
   + x3_is_in_node_11 + x4_is_in_node_11 + x5_is_in_node_11
   + x6_is_in_node_11 + x7_is_in_node_11 + x8_is_in_node_11
   + x9_is_in_node_11 + x10_is_in_node_11 + x11_is_in_node_11
   + x12_is_in_node_11 + x13_is_in_node_11 + x14_is_in_node_11
   + x15_is_in_node_11 + x16_is_in_node_11 + x17_is_in_node_11
   + x18_is_in_node_11 + x19_is_in_node_11 + x20_is_in_node_11
   + x21_is_in_node_11 + x22_is_in_node_11 + x23_is_in_node_11
   + x24_is_in_node_11 + x25_is_in_node_11 + x26_is_in_node_11
   + x27_is_in_node_11 + x28_is_in_node_11 + x29_is_in_node_11
   + x30_is_in_node_11 + x31_is_in_node_11 + x32_is_in_node_11
   + x33_is_in_node_11 + x34_is_in_node_11 + x35_is_in_node_11
   + x36_is_in_node_11 + x37_is_in_node_11 + x38_is_in_node_11
   + x39_is_in_node_11 + x40_is_in_node_11 + x41_is_in_node_11
   + x42_is_in_node_11 + x43_is_in_node_11 + x44_is_in_node_11
   + x45_is_in_node_11 + x46_is_in_node_11 + x47_is_in_node_11
   + x48_is_in_node_11 + x49_is_in_node_11 + x50_is_in_node_11
   + x51_is_in_node_11 + x52_is_in_node_11 + x53_is_in_node_11
   + x54_is_in_node_11 + x55_is_in_node_11 + x56_is_in_node_11
   + x57_is_in_node_11 + x58_is_in_node_11 + x59_is_in_node_11
   + x60_is_in_node_11 + x61_is_in_node_11 + x62_is_in_node_11
   + x63_is_in_node_11 + x64_is_in_node_11 + x65_is_in_node_11
   + x66_is_in_node_11 + x67_is_in_node_11 + x68_is_in_node_11
   + x69_is_in_node_11 + x70_is_in_node_11 + x71_is_in_node_11
   + x72_is_in_node_11 + x73_is_in_node_11 + x74_is_in_node_11
   + x75_is_in_node_11 + x76_is_in_node_11 + x77_is_in_node_11
   + x78_is_in_node_11 + x79_is_in_node_11 + x80_is_in_node_11
   + x81_is_in_node_11 + x82_is_in_node_11 + x83_is_in_node_11
   + x84_is_in_node_11 + x85_is_in_node_11 + x86_is_in_node_11
   + x87_is_in_node_11 + x88_is_in_node_11 + x89_is_in_node_11
   + x90_is_in_node_11 + x91_is_in_node_11 + x92_is_in_node_11
   + x93_is_in_node_11 + x94_is_in_node_11 + x95_is_in_node_11
   + x96_is_in_node_11 + x97_is_in_node_11 + x98_is_in_node_11
   + x99_is_in_node_11 + x100_is_in_node_11 + x101_is_in_node_11
   + x102_is_in_node_11 + x103_is_in_node_11 + x104_is_in_node_11
   + x105_is_in_node_11 + x106_is_in_node_11 + x107_is_in_node_11
   + x108_is_in_node_11 + x109_is_in_node_11 + x110_is_in_node_11
   + x111_is_in_node_11 + x112_is_in_node_11 + x113_is_in_node_11
   + x114_is_in_node_11 + x115_is_in_node_11 + x116_is_in_node_11
   + x117_is_in_node_11 + x118_is_in_node_11 + x119_is_in_node_11
   + x120_is_in_node_11 + x121_is_in_node_11 + x122_is_in_node_11
   + x123_is_in_node_11 + x124_is_in_node_11 + x125_is_in_node_11
   + x126_is_in_node_11 + x127_is_in_node_11 + x128_is_in_node_11
   + x129_is_in_node_11 + x130_is_in_node_11 + x131_is_in_node_11
   + x132_is_in_node_11 + x133_is_in_node_11 + x134_is_in_node_11
   + x135_is_in_node_11 + x136_is_in_node_11 + x137_is_in_node_11
   + x138_is_in_node_11 + x139_is_in_node_11 + x140_is_in_node_11
   + x141_is_in_node_11 + x142_is_in_node_11 + x143_is_in_node_11
   + x144_is_in_node_11 + x145_is_in_node_11 + x146_is_in_node_11
   + x147_is_in_node_11 + x148_is_in_node_11 + x149_is_in_node_11 = 1
 R628: - node_12_contains_any_point + x0_is_in_node_12 <= 0
 R629: - node_12_contains_any_point + x1_is_in_node_12 <= 0
 R630: - node_12_contains_any_point + x2_is_in_node_12 <= 0
 R631: - node_12_contains_any_point + x3_is_in_node_12 <= 0
 R632: - node_12_contains_any_point + x4_is_in_node_12 <= 0
 R633: - node_12_contains_any_point + x5_is_in_node_12 <= 0
 R634: - node_12_contains_any_point + x6_is_in_node_12 <= 0
 R635: - node_12_contains_any_point + x7_is_in_node_12 <= 0
 R636: - node_12_contains_any_point + x8_is_in_node_12 <= 0
 R637: - node_12_contains_any_point + x9_is_in_node_12 <= 0
 R638: - node_12_contains_any_point + x10_is_in_node_12 <= 0
 R639: - node_12_contains_any_point + x11_is_in_node_12 <= 0
 R640: - node_12_contains_any_point + x12_is_in_node_12 <= 0
 R641: - node_12_contains_any_point + x13_is_in_node_12 <= 0
 R642: - node_12_contains_any_point + x14_is_in_node_12 <= 0
 R643: - node_12_contains_any_point + x15_is_in_node_12 <= 0
 R644: - node_12_contains_any_point + x16_is_in_node_12 <= 0
 R645: - node_12_contains_any_point + x17_is_in_node_12 <= 0
 R646: - node_12_contains_any_point + x18_is_in_node_12 <= 0
 R647: - node_12_contains_any_point + x19_is_in_node_12 <= 0
 R648: - node_12_contains_any_point + x20_is_in_node_12 <= 0
 R649: - node_12_contains_any_point + x21_is_in_node_12 <= 0
 R650: - node_12_contains_any_point + x22_is_in_node_12 <= 0
 R651: - node_12_contains_any_point + x23_is_in_node_12 <= 0
 R652: - node_12_contains_any_point + x24_is_in_node_12 <= 0
 R653: - node_12_contains_any_point + x25_is_in_node_12 <= 0
 R654: - node_12_contains_any_point + x26_is_in_node_12 <= 0
 R655: - node_12_contains_any_point + x27_is_in_node_12 <= 0
 R656: - node_12_contains_any_point + x28_is_in_node_12 <= 0
 R657: - node_12_contains_any_point + x29_is_in_node_12 <= 0
 R658: - node_12_contains_any_point + x30_is_in_node_12 <= 0
 R659: - node_12_contains_any_point + x31_is_in_node_12 <= 0
 R660: - node_12_contains_any_point + x32_is_in_node_12 <= 0
 R661: - node_12_contains_any_point + x33_is_in_node_12 <= 0
 R662: - node_12_contains_any_point + x34_is_in_node_12 <= 0
 R663: - node_12_contains_any_point + x35_is_in_node_12 <= 0
 R664: - node_12_contains_any_point + x36_is_in_node_12 <= 0
 R665: - node_12_contains_any_point + x37_is_in_node_12 <= 0
 R666: - node_12_contains_any_point + x38_is_in_node_12 <= 0
 R667: - node_12_contains_any_point + x39_is_in_node_12 <= 0
 R668: - node_12_contains_any_point + x40_is_in_node_12 <= 0
 R669: - node_12_contains_any_point + x41_is_in_node_12 <= 0
 R670: - node_12_contains_any_point + x42_is_in_node_12 <= 0
 R671: - node_12_contains_any_point + x43_is_in_node_12 <= 0
 R672: - node_12_contains_any_point + x44_is_in_node_12 <= 0
 R673: - node_12_contains_any_point + x45_is_in_node_12 <= 0
 R674: - node_12_contains_any_point + x46_is_in_node_12 <= 0
 R675: - node_12_contains_any_point + x47_is_in_node_12 <= 0
 R676: - node_12_contains_any_point + x48_is_in_node_12 <= 0
 R677: - node_12_contains_any_point + x49_is_in_node_12 <= 0
 R678: - node_12_contains_any_point + x50_is_in_node_12 <= 0
 R679: - node_12_contains_any_point + x51_is_in_node_12 <= 0
 R680: - node_12_contains_any_point + x52_is_in_node_12 <= 0
 R681: - node_12_contains_any_point + x53_is_in_node_12 <= 0
 R682: - node_12_contains_any_point + x54_is_in_node_12 <= 0
 R683: - node_12_contains_any_point + x55_is_in_node_12 <= 0
 R684: - node_12_contains_any_point + x56_is_in_node_12 <= 0
 R685: - node_12_contains_any_point + x57_is_in_node_12 <= 0
 R686: - node_12_contains_any_point + x58_is_in_node_12 <= 0
 R687: - node_12_contains_any_point + x59_is_in_node_12 <= 0
 R688: - node_12_contains_any_point + x60_is_in_node_12 <= 0
 R689: - node_12_contains_any_point + x61_is_in_node_12 <= 0
 R690: - node_12_contains_any_point + x62_is_in_node_12 <= 0
 R691: - node_12_contains_any_point + x63_is_in_node_12 <= 0
 R692: - node_12_contains_any_point + x64_is_in_node_12 <= 0
 R693: - node_12_contains_any_point + x65_is_in_node_12 <= 0
 R694: - node_12_contains_any_point + x66_is_in_node_12 <= 0
 R695: - node_12_contains_any_point + x67_is_in_node_12 <= 0
 R696: - node_12_contains_any_point + x68_is_in_node_12 <= 0
 R697: - node_12_contains_any_point + x69_is_in_node_12 <= 0
 R698: - node_12_contains_any_point + x70_is_in_node_12 <= 0
 R699: - node_12_contains_any_point + x71_is_in_node_12 <= 0
 R700: - node_12_contains_any_point + x72_is_in_node_12 <= 0
 R701: - node_12_contains_any_point + x73_is_in_node_12 <= 0
 R702: - node_12_contains_any_point + x74_is_in_node_12 <= 0
 R703: - node_12_contains_any_point + x75_is_in_node_12 <= 0
 R704: - node_12_contains_any_point + x76_is_in_node_12 <= 0
 R705: - node_12_contains_any_point + x77_is_in_node_12 <= 0
 R706: - node_12_contains_any_point + x78_is_in_node_12 <= 0
 R707: - node_12_contains_any_point + x79_is_in_node_12 <= 0
 R708: - node_12_contains_any_point + x80_is_in_node_12 <= 0
 R709: - node_12_contains_any_point + x81_is_in_node_12 <= 0
 R710: - node_12_contains_any_point + x82_is_in_node_12 <= 0
 R711: - node_12_contains_any_point + x83_is_in_node_12 <= 0
 R712: - node_12_contains_any_point + x84_is_in_node_12 <= 0
 R713: - node_12_contains_any_point + x85_is_in_node_12 <= 0
 R714: - node_12_contains_any_point + x86_is_in_node_12 <= 0
 R715: - node_12_contains_any_point + x87_is_in_node_12 <= 0
 R716: - node_12_contains_any_point + x88_is_in_node_12 <= 0
 R717: - node_12_contains_any_point + x89_is_in_node_12 <= 0
 R718: - node_12_contains_any_point + x90_is_in_node_12 <= 0
 R719: - node_12_contains_any_point + x91_is_in_node_12 <= 0
 R720: - node_12_contains_any_point + x92_is_in_node_12 <= 0
 R721: - node_12_contains_any_point + x93_is_in_node_12 <= 0
 R722: - node_12_contains_any_point + x94_is_in_node_12 <= 0
 R723: - node_12_contains_any_point + x95_is_in_node_12 <= 0
 R724: - node_12_contains_any_point + x96_is_in_node_12 <= 0
 R725: - node_12_contains_any_point + x97_is_in_node_12 <= 0
 R726: - node_12_contains_any_point + x98_is_in_node_12 <= 0
 R727: - node_12_contains_any_point + x99_is_in_node_12 <= 0
 R728: - node_12_contains_any_point + x100_is_in_node_12 <= 0
 R729: - node_12_contains_any_point + x101_is_in_node_12 <= 0
 R730: - node_12_contains_any_point + x102_is_in_node_12 <= 0
 R731: - node_12_contains_any_point + x103_is_in_node_12 <= 0
 R732: - node_12_contains_any_point + x104_is_in_node_12 <= 0
 R733: - node_12_contains_any_point + x105_is_in_node_12 <= 0
 R734: - node_12_contains_any_point + x106_is_in_node_12 <= 0
 R735: - node_12_contains_any_point + x107_is_in_node_12 <= 0
 R736: - node_12_contains_any_point + x108_is_in_node_12 <= 0
 R737: - node_12_contains_any_point + x109_is_in_node_12 <= 0
 R738: - node_12_contains_any_point + x110_is_in_node_12 <= 0
 R739: - node_12_contains_any_point + x111_is_in_node_12 <= 0
 R740: - node_12_contains_any_point + x112_is_in_node_12 <= 0
 R741: - node_12_contains_any_point + x113_is_in_node_12 <= 0
 R742: - node_12_contains_any_point + x114_is_in_node_12 <= 0
 R743: - node_12_contains_any_point + x115_is_in_node_12 <= 0
 R744: - node_12_contains_any_point + x116_is_in_node_12 <= 0
 R745: - node_12_contains_any_point + x117_is_in_node_12 <= 0
 R746: - node_12_contains_any_point + x118_is_in_node_12 <= 0
 R747: - node_12_contains_any_point + x119_is_in_node_12 <= 0
 R748: - node_12_contains_any_point + x120_is_in_node_12 <= 0
 R749: - node_12_contains_any_point + x121_is_in_node_12 <= 0
 R750: - node_12_contains_any_point + x122_is_in_node_12 <= 0
 R751: - node_12_contains_any_point + x123_is_in_node_12 <= 0
 R752: - node_12_contains_any_point + x124_is_in_node_12 <= 0
 R753: - node_12_contains_any_point + x125_is_in_node_12 <= 0
 R754: - node_12_contains_any_point + x126_is_in_node_12 <= 0
 R755: - node_12_contains_any_point + x127_is_in_node_12 <= 0
 R756: - node_12_contains_any_point + x128_is_in_node_12 <= 0
 R757: - node_12_contains_any_point + x129_is_in_node_12 <= 0
 R758: - node_12_contains_any_point + x130_is_in_node_12 <= 0
 R759: - node_12_contains_any_point + x131_is_in_node_12 <= 0
 R760: - node_12_contains_any_point + x132_is_in_node_12 <= 0
 R761: - node_12_contains_any_point + x133_is_in_node_12 <= 0
 R762: - node_12_contains_any_point + x134_is_in_node_12 <= 0
 R763: - node_12_contains_any_point + x135_is_in_node_12 <= 0
 R764: - node_12_contains_any_point + x136_is_in_node_12 <= 0
 R765: - node_12_contains_any_point + x137_is_in_node_12 <= 0
 R766: - node_12_contains_any_point + x138_is_in_node_12 <= 0
 R767: - node_12_contains_any_point + x139_is_in_node_12 <= 0
 R768: - node_12_contains_any_point + x140_is_in_node_12 <= 0
 R769: - node_12_contains_any_point + x141_is_in_node_12 <= 0
 R770: - node_12_contains_any_point + x142_is_in_node_12 <= 0
 R771: - node_12_contains_any_point + x143_is_in_node_12 <= 0
 R772: - node_12_contains_any_point + x144_is_in_node_12 <= 0
 R773: - node_12_contains_any_point + x145_is_in_node_12 <= 0
 R774: - node_12_contains_any_point + x146_is_in_node_12 <= 0
 R775: - node_12_contains_any_point + x147_is_in_node_12 <= 0
 R776: - node_12_contains_any_point + x148_is_in_node_12 <= 0
 R777: - node_12_contains_any_point + x149_is_in_node_12 <= 0
 R778: - 7 node_12_contains_any_point + x0_is_in_node_12 + x1_is_in_node_12
   + x2_is_in_node_12 + x3_is_in_node_12 + x4_is_in_node_12
   + x5_is_in_node_12 + x6_is_in_node_12 + x7_is_in_node_12
   + x8_is_in_node_12 + x9_is_in_node_12 + x10_is_in_node_12
   + x11_is_in_node_12 + x12_is_in_node_12 + x13_is_in_node_12
   + x14_is_in_node_12 + x15_is_in_node_12 + x16_is_in_node_12
   + x17_is_in_node_12 + x18_is_in_node_12 + x19_is_in_node_12
   + x20_is_in_node_12 + x21_is_in_node_12 + x22_is_in_node_12
   + x23_is_in_node_12 + x24_is_in_node_12 + x25_is_in_node_12
   + x26_is_in_node_12 + x27_is_in_node_12 + x28_is_in_node_12
   + x29_is_in_node_12 + x30_is_in_node_12 + x31_is_in_node_12
   + x32_is_in_node_12 + x33_is_in_node_12 + x34_is_in_node_12
   + x35_is_in_node_12 + x36_is_in_node_12 + x37_is_in_node_12
   + x38_is_in_node_12 + x39_is_in_node_12 + x40_is_in_node_12
   + x41_is_in_node_12 + x42_is_in_node_12 + x43_is_in_node_12
   + x44_is_in_node_12 + x45_is_in_node_12 + x46_is_in_node_12
   + x47_is_in_node_12 + x48_is_in_node_12 + x49_is_in_node_12
   + x50_is_in_node_12 + x51_is_in_node_12 + x52_is_in_node_12
   + x53_is_in_node_12 + x54_is_in_node_12 + x55_is_in_node_12
   + x56_is_in_node_12 + x57_is_in_node_12 + x58_is_in_node_12
   + x59_is_in_node_12 + x60_is_in_node_12 + x61_is_in_node_12
   + x62_is_in_node_12 + x63_is_in_node_12 + x64_is_in_node_12
   + x65_is_in_node_12 + x66_is_in_node_12 + x67_is_in_node_12
   + x68_is_in_node_12 + x69_is_in_node_12 + x70_is_in_node_12
   + x71_is_in_node_12 + x72_is_in_node_12 + x73_is_in_node_12
   + x74_is_in_node_12 + x75_is_in_node_12 + x76_is_in_node_12
   + x77_is_in_node_12 + x78_is_in_node_12 + x79_is_in_node_12
   + x80_is_in_node_12 + x81_is_in_node_12 + x82_is_in_node_12
   + x83_is_in_node_12 + x84_is_in_node_12 + x85_is_in_node_12
   + x86_is_in_node_12 + x87_is_in_node_12 + x88_is_in_node_12
   + x89_is_in_node_12 + x90_is_in_node_12 + x91_is_in_node_12
   + x92_is_in_node_12 + x93_is_in_node_12 + x94_is_in_node_12
   + x95_is_in_node_12 + x96_is_in_node_12 + x97_is_in_node_12
   + x98_is_in_node_12 + x99_is_in_node_12 + x100_is_in_node_12
   + x101_is_in_node_12 + x102_is_in_node_12 + x103_is_in_node_12
   + x104_is_in_node_12 + x105_is_in_node_12 + x106_is_in_node_12
   + x107_is_in_node_12 + x108_is_in_node_12 + x109_is_in_node_12
   + x110_is_in_node_12 + x111_is_in_node_12 + x112_is_in_node_12
   + x113_is_in_node_12 + x114_is_in_node_12 + x115_is_in_node_12
   + x116_is_in_node_12 + x117_is_in_node_12 + x118_is_in_node_12
   + x119_is_in_node_12 + x120_is_in_node_12 + x121_is_in_node_12
   + x122_is_in_node_12 + x123_is_in_node_12 + x124_is_in_node_12
   + x125_is_in_node_12 + x126_is_in_node_12 + x127_is_in_node_12
   + x128_is_in_node_12 + x129_is_in_node_12 + x130_is_in_node_12
   + x131_is_in_node_12 + x132_is_in_node_12 + x133_is_in_node_12
   + x134_is_in_node_12 + x135_is_in_node_12 + x136_is_in_node_12
   + x137_is_in_node_12 + x138_is_in_node_12 + x139_is_in_node_12
   + x140_is_in_node_12 + x141_is_in_node_12 + x142_is_in_node_12
   + x143_is_in_node_12 + x144_is_in_node_12 + x145_is_in_node_12
   + x146_is_in_node_12 + x147_is_in_node_12 + x148_is_in_node_12
   + x149_is_in_node_12 >= 0
 R779: x0_is_in_node_12 + x1_is_in_node_12 + x2_is_in_node_12
   + x3_is_in_node_12 + x4_is_in_node_12 + x5_is_in_node_12
   + x6_is_in_node_12 + x7_is_in_node_12 + x8_is_in_node_12
   + x9_is_in_node_12 + x10_is_in_node_12 + x11_is_in_node_12
   + x12_is_in_node_12 + x13_is_in_node_12 + x14_is_in_node_12
   + x15_is_in_node_12 + x16_is_in_node_12 + x17_is_in_node_12
   + x18_is_in_node_12 + x19_is_in_node_12 + x20_is_in_node_12
   + x21_is_in_node_12 + x22_is_in_node_12 + x23_is_in_node_12
   + x24_is_in_node_12 + x25_is_in_node_12 + x26_is_in_node_12
   + x27_is_in_node_12 + x28_is_in_node_12 + x29_is_in_node_12
   + x30_is_in_node_12 + x31_is_in_node_12 + x32_is_in_node_12
   + x33_is_in_node_12 + x34_is_in_node_12 + x35_is_in_node_12
   + x36_is_in_node_12 + x37_is_in_node_12 + x38_is_in_node_12
   + x39_is_in_node_12 + x40_is_in_node_12 + x41_is_in_node_12
   + x42_is_in_node_12 + x43_is_in_node_12 + x44_is_in_node_12
   + x45_is_in_node_12 + x46_is_in_node_12 + x47_is_in_node_12
   + x48_is_in_node_12 + x49_is_in_node_12 + x50_is_in_node_12
   + x51_is_in_node_12 + x52_is_in_node_12 + x53_is_in_node_12
   + x54_is_in_node_12 + x55_is_in_node_12 + x56_is_in_node_12
   + x57_is_in_node_12 + x58_is_in_node_12 + x59_is_in_node_12
   + x60_is_in_node_12 + x61_is_in_node_12 + x62_is_in_node_12
   + x63_is_in_node_12 + x64_is_in_node_12 + x65_is_in_node_12
   + x66_is_in_node_12 + x67_is_in_node_12 + x68_is_in_node_12
   + x69_is_in_node_12 + x70_is_in_node_12 + x71_is_in_node_12
   + x72_is_in_node_12 + x73_is_in_node_12 + x74_is_in_node_12
   + x75_is_in_node_12 + x76_is_in_node_12 + x77_is_in_node_12
   + x78_is_in_node_12 + x79_is_in_node_12 + x80_is_in_node_12
   + x81_is_in_node_12 + x82_is_in_node_12 + x83_is_in_node_12
   + x84_is_in_node_12 + x85_is_in_node_12 + x86_is_in_node_12
   + x87_is_in_node_12 + x88_is_in_node_12 + x89_is_in_node_12
   + x90_is_in_node_12 + x91_is_in_node_12 + x92_is_in_node_12
   + x93_is_in_node_12 + x94_is_in_node_12 + x95_is_in_node_12
   + x96_is_in_node_12 + x97_is_in_node_12 + x98_is_in_node_12
   + x99_is_in_node_12 + x100_is_in_node_12 + x101_is_in_node_12
   + x102_is_in_node_12 + x103_is_in_node_12 + x104_is_in_node_12
   + x105_is_in_node_12 + x106_is_in_node_12 + x107_is_in_node_12
   + x108_is_in_node_12 + x109_is_in_node_12 + x110_is_in_node_12
   + x111_is_in_node_12 + x112_is_in_node_12 + x113_is_in_node_12
   + x114_is_in_node_12 + x115_is_in_node_12 + x116_is_in_node_12
   + x117_is_in_node_12 + x118_is_in_node_12 + x119_is_in_node_12
   + x120_is_in_node_12 + x121_is_in_node_12 + x122_is_in_node_12
   + x123_is_in_node_12 + x124_is_in_node_12 + x125_is_in_node_12
   + x126_is_in_node_12 + x127_is_in_node_12 + x128_is_in_node_12
   + x129_is_in_node_12 + x130_is_in_node_12 + x131_is_in_node_12
   + x132_is_in_node_12 + x133_is_in_node_12 + x134_is_in_node_12
   + x135_is_in_node_12 + x136_is_in_node_12 + x137_is_in_node_12
   + x138_is_in_node_12 + x139_is_in_node_12 + x140_is_in_node_12
   + x141_is_in_node_12 + x142_is_in_node_12 + x143_is_in_node_12
   + x144_is_in_node_12 + x145_is_in_node_12 + x146_is_in_node_12
   + x147_is_in_node_12 + x148_is_in_node_12 + x149_is_in_node_12 = 1
 R780: - node_13_contains_any_point + x0_is_in_node_13 <= 0
 R781: - node_13_contains_any_point + x1_is_in_node_13 <= 0
 R782: - node_13_contains_any_point + x2_is_in_node_13 <= 0
 R783: - node_13_contains_any_point + x3_is_in_node_13 <= 0
 R784: - node_13_contains_any_point + x4_is_in_node_13 <= 0
 R785: - node_13_contains_any_point + x5_is_in_node_13 <= 0
 R786: - node_13_contains_any_point + x6_is_in_node_13 <= 0
 R787: - node_13_contains_any_point + x7_is_in_node_13 <= 0
 R788: - node_13_contains_any_point + x8_is_in_node_13 <= 0
 R789: - node_13_contains_any_point + x9_is_in_node_13 <= 0
 R790: - node_13_contains_any_point + x10_is_in_node_13 <= 0
 R791: - node_13_contains_any_point + x11_is_in_node_13 <= 0
 R792: - node_13_contains_any_point + x12_is_in_node_13 <= 0
 R793: - node_13_contains_any_point + x13_is_in_node_13 <= 0
 R794: - node_13_contains_any_point + x14_is_in_node_13 <= 0
 R795: - node_13_contains_any_point + x15_is_in_node_13 <= 0
 R796: - node_13_contains_any_point + x16_is_in_node_13 <= 0
 R797: - node_13_contains_any_point + x17_is_in_node_13 <= 0
 R798: - node_13_contains_any_point + x18_is_in_node_13 <= 0
 R799: - node_13_contains_any_point + x19_is_in_node_13 <= 0
 R800: - node_13_contains_any_point + x20_is_in_node_13 <= 0
 R801: - node_13_contains_any_point + x21_is_in_node_13 <= 0
 R802: - node_13_contains_any_point + x22_is_in_node_13 <= 0
 R803: - node_13_contains_any_point + x23_is_in_node_13 <= 0
 R804: - node_13_contains_any_point + x24_is_in_node_13 <= 0
 R805: - node_13_contains_any_point + x25_is_in_node_13 <= 0
 R806: - node_13_contains_any_point + x26_is_in_node_13 <= 0
 R807: - node_13_contains_any_point + x27_is_in_node_13 <= 0
 R808: - node_13_contains_any_point + x28_is_in_node_13 <= 0
 R809: - node_13_contains_any_point + x29_is_in_node_13 <= 0
 R810: - node_13_contains_any_point + x30_is_in_node_13 <= 0
 R811: - node_13_contains_any_point + x31_is_in_node_13 <= 0
 R812: - node_13_contains_any_point + x32_is_in_node_13 <= 0
 R813: - node_13_contains_any_point + x33_is_in_node_13 <= 0
 R814: - node_13_contains_any_point + x34_is_in_node_13 <= 0
 R815: - node_13_contains_any_point + x35_is_in_node_13 <= 0
 R816: - node_13_contains_any_point + x36_is_in_node_13 <= 0
 R817: - node_13_contains_any_point + x37_is_in_node_13 <= 0
 R818: - node_13_contains_any_point + x38_is_in_node_13 <= 0
 R819: - node_13_contains_any_point + x39_is_in_node_13 <= 0
 R820: - node_13_contains_any_point + x40_is_in_node_13 <= 0
 R821: - node_13_contains_any_point + x41_is_in_node_13 <= 0
 R822: - node_13_contains_any_point + x42_is_in_node_13 <= 0
 R823: - node_13_contains_any_point + x43_is_in_node_13 <= 0
 R824: - node_13_contains_any_point + x44_is_in_node_13 <= 0
 R825: - node_13_contains_any_point + x45_is_in_node_13 <= 0
 R826: - node_13_contains_any_point + x46_is_in_node_13 <= 0
 R827: - node_13_contains_any_point + x47_is_in_node_13 <= 0
 R828: - node_13_contains_any_point + x48_is_in_node_13 <= 0
 R829: - node_13_contains_any_point + x49_is_in_node_13 <= 0
 R830: - node_13_contains_any_point + x50_is_in_node_13 <= 0
 R831: - node_13_contains_any_point + x51_is_in_node_13 <= 0
 R832: - node_13_contains_any_point + x52_is_in_node_13 <= 0
 R833: - node_13_contains_any_point + x53_is_in_node_13 <= 0
 R834: - node_13_contains_any_point + x54_is_in_node_13 <= 0
 R835: - node_13_contains_any_point + x55_is_in_node_13 <= 0
 R836: - node_13_contains_any_point + x56_is_in_node_13 <= 0
 R837: - node_13_contains_any_point + x57_is_in_node_13 <= 0
 R838: - node_13_contains_any_point + x58_is_in_node_13 <= 0
 R839: - node_13_contains_any_point + x59_is_in_node_13 <= 0
 R840: - node_13_contains_any_point + x60_is_in_node_13 <= 0
 R841: - node_13_contains_any_point + x61_is_in_node_13 <= 0
 R842: - node_13_contains_any_point + x62_is_in_node_13 <= 0
 R843: - node_13_contains_any_point + x63_is_in_node_13 <= 0
 R844: - node_13_contains_any_point + x64_is_in_node_13 <= 0
 R845: - node_13_contains_any_point + x65_is_in_node_13 <= 0
 R846: - node_13_contains_any_point + x66_is_in_node_13 <= 0
 R847: - node_13_contains_any_point + x67_is_in_node_13 <= 0
 R848: - node_13_contains_any_point + x68_is_in_node_13 <= 0
 R849: - node_13_contains_any_point + x69_is_in_node_13 <= 0
 R850: - node_13_contains_any_point + x70_is_in_node_13 <= 0
 R851: - node_13_contains_any_point + x71_is_in_node_13 <= 0
 R852: - node_13_contains_any_point + x72_is_in_node_13 <= 0
 R853: - node_13_contains_any_point + x73_is_in_node_13 <= 0
 R854: - node_13_contains_any_point + x74_is_in_node_13 <= 0
 R855: - node_13_contains_any_point + x75_is_in_node_13 <= 0
 R856: - node_13_contains_any_point + x76_is_in_node_13 <= 0
 R857: - node_13_contains_any_point + x77_is_in_node_13 <= 0
 R858: - node_13_contains_any_point + x78_is_in_node_13 <= 0
 R859: - node_13_contains_any_point + x79_is_in_node_13 <= 0
 R860: - node_13_contains_any_point + x80_is_in_node_13 <= 0
 R861: - node_13_contains_any_point + x81_is_in_node_13 <= 0
 R862: - node_13_contains_any_point + x82_is_in_node_13 <= 0
 R863: - node_13_contains_any_point + x83_is_in_node_13 <= 0
 R864: - node_13_contains_any_point + x84_is_in_node_13 <= 0
 R865: - node_13_contains_any_point + x85_is_in_node_13 <= 0
 R866: - node_13_contains_any_point + x86_is_in_node_13 <= 0
 R867: - node_13_contains_any_point + x87_is_in_node_13 <= 0
 R868: - node_13_contains_any_point + x88_is_in_node_13 <= 0
 R869: - node_13_contains_any_point + x89_is_in_node_13 <= 0
 R870: - node_13_contains_any_point + x90_is_in_node_13 <= 0
 R871: - node_13_contains_any_point + x91_is_in_node_13 <= 0
 R872: - node_13_contains_any_point + x92_is_in_node_13 <= 0
 R873: - node_13_contains_any_point + x93_is_in_node_13 <= 0
 R874: - node_13_contains_any_point + x94_is_in_node_13 <= 0
 R875: - node_13_contains_any_point + x95_is_in_node_13 <= 0
 R876: - node_13_contains_any_point + x96_is_in_node_13 <= 0
 R877: - node_13_contains_any_point + x97_is_in_node_13 <= 0
 R878: - node_13_contains_any_point + x98_is_in_node_13 <= 0
 R879: - node_13_contains_any_point + x99_is_in_node_13 <= 0
 R880: - node_13_contains_any_point + x100_is_in_node_13 <= 0
 R881: - node_13_contains_any_point + x101_is_in_node_13 <= 0
 R882: - node_13_contains_any_point + x102_is_in_node_13 <= 0
 R883: - node_13_contains_any_point + x103_is_in_node_13 <= 0
 R884: - node_13_contains_any_point + x104_is_in_node_13 <= 0
 R885: - node_13_contains_any_point + x105_is_in_node_13 <= 0
 R886: - node_13_contains_any_point + x106_is_in_node_13 <= 0
 R887: - node_13_contains_any_point + x107_is_in_node_13 <= 0
 R888: - node_13_contains_any_point + x108_is_in_node_13 <= 0
 R889: - node_13_contains_any_point + x109_is_in_node_13 <= 0
 R890: - node_13_contains_any_point + x110_is_in_node_13 <= 0
 R891: - node_13_contains_any_point + x111_is_in_node_13 <= 0
 R892: - node_13_contains_any_point + x112_is_in_node_13 <= 0
 R893: - node_13_contains_any_point + x113_is_in_node_13 <= 0
 R894: - node_13_contains_any_point + x114_is_in_node_13 <= 0
 R895: - node_13_contains_any_point + x115_is_in_node_13 <= 0
 R896: - node_13_contains_any_point + x116_is_in_node_13 <= 0
 R897: - node_13_contains_any_point + x117_is_in_node_13 <= 0
 R898: - node_13_contains_any_point + x118_is_in_node_13 <= 0
 R899: - node_13_contains_any_point + x119_is_in_node_13 <= 0
 R900: - node_13_contains_any_point + x120_is_in_node_13 <= 0
 R901: - node_13_contains_any_point + x121_is_in_node_13 <= 0
 R902: - node_13_contains_any_point + x122_is_in_node_13 <= 0
 R903: - node_13_contains_any_point + x123_is_in_node_13 <= 0
 R904: - node_13_contains_any_point + x124_is_in_node_13 <= 0
 R905: - node_13_contains_any_point + x125_is_in_node_13 <= 0
 R906: - node_13_contains_any_point + x126_is_in_node_13 <= 0
 R907: - node_13_contains_any_point + x127_is_in_node_13 <= 0
 R908: - node_13_contains_any_point + x128_is_in_node_13 <= 0
 R909: - node_13_contains_any_point + x129_is_in_node_13 <= 0
 R910: - node_13_contains_any_point + x130_is_in_node_13 <= 0
 R911: - node_13_contains_any_point + x131_is_in_node_13 <= 0
 R912: - node_13_contains_any_point + x132_is_in_node_13 <= 0
 R913: - node_13_contains_any_point + x133_is_in_node_13 <= 0
 R914: - node_13_contains_any_point + x134_is_in_node_13 <= 0
 R915: - node_13_contains_any_point + x135_is_in_node_13 <= 0
 R916: - node_13_contains_any_point + x136_is_in_node_13 <= 0
 R917: - node_13_contains_any_point + x137_is_in_node_13 <= 0
 R918: - node_13_contains_any_point + x138_is_in_node_13 <= 0
 R919: - node_13_contains_any_point + x139_is_in_node_13 <= 0
 R920: - node_13_contains_any_point + x140_is_in_node_13 <= 0
 R921: - node_13_contains_any_point + x141_is_in_node_13 <= 0
 R922: - node_13_contains_any_point + x142_is_in_node_13 <= 0
 R923: - node_13_contains_any_point + x143_is_in_node_13 <= 0
 R924: - node_13_contains_any_point + x144_is_in_node_13 <= 0
 R925: - node_13_contains_any_point + x145_is_in_node_13 <= 0
 R926: - node_13_contains_any_point + x146_is_in_node_13 <= 0
 R927: - node_13_contains_any_point + x147_is_in_node_13 <= 0
 R928: - node_13_contains_any_point + x148_is_in_node_13 <= 0
 R929: - node_13_contains_any_point + x149_is_in_node_13 <= 0
 R930: - 7 node_13_contains_any_point + x0_is_in_node_13 + x1_is_in_node_13
   + x2_is_in_node_13 + x3_is_in_node_13 + x4_is_in_node_13
   + x5_is_in_node_13 + x6_is_in_node_13 + x7_is_in_node_13
   + x8_is_in_node_13 + x9_is_in_node_13 + x10_is_in_node_13
   + x11_is_in_node_13 + x12_is_in_node_13 + x13_is_in_node_13
   + x14_is_in_node_13 + x15_is_in_node_13 + x16_is_in_node_13
   + x17_is_in_node_13 + x18_is_in_node_13 + x19_is_in_node_13
   + x20_is_in_node_13 + x21_is_in_node_13 + x22_is_in_node_13
   + x23_is_in_node_13 + x24_is_in_node_13 + x25_is_in_node_13
   + x26_is_in_node_13 + x27_is_in_node_13 + x28_is_in_node_13
   + x29_is_in_node_13 + x30_is_in_node_13 + x31_is_in_node_13
   + x32_is_in_node_13 + x33_is_in_node_13 + x34_is_in_node_13
   + x35_is_in_node_13 + x36_is_in_node_13 + x37_is_in_node_13
   + x38_is_in_node_13 + x39_is_in_node_13 + x40_is_in_node_13
   + x41_is_in_node_13 + x42_is_in_node_13 + x43_is_in_node_13
   + x44_is_in_node_13 + x45_is_in_node_13 + x46_is_in_node_13
   + x47_is_in_node_13 + x48_is_in_node_13 + x49_is_in_node_13
   + x50_is_in_node_13 + x51_is_in_node_13 + x52_is_in_node_13
   + x53_is_in_node_13 + x54_is_in_node_13 + x55_is_in_node_13
   + x56_is_in_node_13 + x57_is_in_node_13 + x58_is_in_node_13
   + x59_is_in_node_13 + x60_is_in_node_13 + x61_is_in_node_13
   + x62_is_in_node_13 + x63_is_in_node_13 + x64_is_in_node_13
   + x65_is_in_node_13 + x66_is_in_node_13 + x67_is_in_node_13
   + x68_is_in_node_13 + x69_is_in_node_13 + x70_is_in_node_13
   + x71_is_in_node_13 + x72_is_in_node_13 + x73_is_in_node_13
   + x74_is_in_node_13 + x75_is_in_node_13 + x76_is_in_node_13
   + x77_is_in_node_13 + x78_is_in_node_13 + x79_is_in_node_13
   + x80_is_in_node_13 + x81_is_in_node_13 + x82_is_in_node_13
   + x83_is_in_node_13 + x84_is_in_node_13 + x85_is_in_node_13
   + x86_is_in_node_13 + x87_is_in_node_13 + x88_is_in_node_13
   + x89_is_in_node_13 + x90_is_in_node_13 + x91_is_in_node_13
   + x92_is_in_node_13 + x93_is_in_node_13 + x94_is_in_node_13
   + x95_is_in_node_13 + x96_is_in_node_13 + x97_is_in_node_13
   + x98_is_in_node_13 + x99_is_in_node_13 + x100_is_in_node_13
   + x101_is_in_node_13 + x102_is_in_node_13 + x103_is_in_node_13
   + x104_is_in_node_13 + x105_is_in_node_13 + x106_is_in_node_13
   + x107_is_in_node_13 + x108_is_in_node_13 + x109_is_in_node_13
   + x110_is_in_node_13 + x111_is_in_node_13 + x112_is_in_node_13
   + x113_is_in_node_13 + x114_is_in_node_13 + x115_is_in_node_13
   + x116_is_in_node_13 + x117_is_in_node_13 + x118_is_in_node_13
   + x119_is_in_node_13 + x120_is_in_node_13 + x121_is_in_node_13
   + x122_is_in_node_13 + x123_is_in_node_13 + x124_is_in_node_13
   + x125_is_in_node_13 + x126_is_in_node_13 + x127_is_in_node_13
   + x128_is_in_node_13 + x129_is_in_node_13 + x130_is_in_node_13
   + x131_is_in_node_13 + x132_is_in_node_13 + x133_is_in_node_13
   + x134_is_in_node_13 + x135_is_in_node_13 + x136_is_in_node_13
   + x137_is_in_node_13 + x138_is_in_node_13 + x139_is_in_node_13
   + x140_is_in_node_13 + x141_is_in_node_13 + x142_is_in_node_13
   + x143_is_in_node_13 + x144_is_in_node_13 + x145_is_in_node_13
   + x146_is_in_node_13 + x147_is_in_node_13 + x148_is_in_node_13
   + x149_is_in_node_13 >= 0
 R931: x0_is_in_node_13 + x1_is_in_node_13 + x2_is_in_node_13
   + x3_is_in_node_13 + x4_is_in_node_13 + x5_is_in_node_13
   + x6_is_in_node_13 + x7_is_in_node_13 + x8_is_in_node_13
   + x9_is_in_node_13 + x10_is_in_node_13 + x11_is_in_node_13
   + x12_is_in_node_13 + x13_is_in_node_13 + x14_is_in_node_13
   + x15_is_in_node_13 + x16_is_in_node_13 + x17_is_in_node_13
   + x18_is_in_node_13 + x19_is_in_node_13 + x20_is_in_node_13
   + x21_is_in_node_13 + x22_is_in_node_13 + x23_is_in_node_13
   + x24_is_in_node_13 + x25_is_in_node_13 + x26_is_in_node_13
   + x27_is_in_node_13 + x28_is_in_node_13 + x29_is_in_node_13
   + x30_is_in_node_13 + x31_is_in_node_13 + x32_is_in_node_13
   + x33_is_in_node_13 + x34_is_in_node_13 + x35_is_in_node_13
   + x36_is_in_node_13 + x37_is_in_node_13 + x38_is_in_node_13
   + x39_is_in_node_13 + x40_is_in_node_13 + x41_is_in_node_13
   + x42_is_in_node_13 + x43_is_in_node_13 + x44_is_in_node_13
   + x45_is_in_node_13 + x46_is_in_node_13 + x47_is_in_node_13
   + x48_is_in_node_13 + x49_is_in_node_13 + x50_is_in_node_13
   + x51_is_in_node_13 + x52_is_in_node_13 + x53_is_in_node_13
   + x54_is_in_node_13 + x55_is_in_node_13 + x56_is_in_node_13
   + x57_is_in_node_13 + x58_is_in_node_13 + x59_is_in_node_13
   + x60_is_in_node_13 + x61_is_in_node_13 + x62_is_in_node_13
   + x63_is_in_node_13 + x64_is_in_node_13 + x65_is_in_node_13
   + x66_is_in_node_13 + x67_is_in_node_13 + x68_is_in_node_13
   + x69_is_in_node_13 + x70_is_in_node_13 + x71_is_in_node_13
   + x72_is_in_node_13 + x73_is_in_node_13 + x74_is_in_node_13
   + x75_is_in_node_13 + x76_is_in_node_13 + x77_is_in_node_13
   + x78_is_in_node_13 + x79_is_in_node_13 + x80_is_in_node_13
   + x81_is_in_node_13 + x82_is_in_node_13 + x83_is_in_node_13
   + x84_is_in_node_13 + x85_is_in_node_13 + x86_is_in_node_13
   + x87_is_in_node_13 + x88_is_in_node_13 + x89_is_in_node_13
   + x90_is_in_node_13 + x91_is_in_node_13 + x92_is_in_node_13
   + x93_is_in_node_13 + x94_is_in_node_13 + x95_is_in_node_13
   + x96_is_in_node_13 + x97_is_in_node_13 + x98_is_in_node_13
   + x99_is_in_node_13 + x100_is_in_node_13 + x101_is_in_node_13
   + x102_is_in_node_13 + x103_is_in_node_13 + x104_is_in_node_13
   + x105_is_in_node_13 + x106_is_in_node_13 + x107_is_in_node_13
   + x108_is_in_node_13 + x109_is_in_node_13 + x110_is_in_node_13
   + x111_is_in_node_13 + x112_is_in_node_13 + x113_is_in_node_13
   + x114_is_in_node_13 + x115_is_in_node_13 + x116_is_in_node_13
   + x117_is_in_node_13 + x118_is_in_node_13 + x119_is_in_node_13
   + x120_is_in_node_13 + x121_is_in_node_13 + x122_is_in_node_13
   + x123_is_in_node_13 + x124_is_in_node_13 + x125_is_in_node_13
   + x126_is_in_node_13 + x127_is_in_node_13 + x128_is_in_node_13
   + x129_is_in_node_13 + x130_is_in_node_13 + x131_is_in_node_13
   + x132_is_in_node_13 + x133_is_in_node_13 + x134_is_in_node_13
   + x135_is_in_node_13 + x136_is_in_node_13 + x137_is_in_node_13
   + x138_is_in_node_13 + x139_is_in_node_13 + x140_is_in_node_13
   + x141_is_in_node_13 + x142_is_in_node_13 + x143_is_in_node_13
   + x144_is_in_node_13 + x145_is_in_node_13 + x146_is_in_node_13
   + x147_is_in_node_13 + x148_is_in_node_13 + x149_is_in_node_13 = 1
 R932: - node_14_contains_any_point + x0_is_in_node_14 <= 0
 R933: - node_14_contains_any_point + x1_is_in_node_14 <= 0
 R934: - node_14_contains_any_point + x2_is_in_node_14 <= 0
 R935: - node_14_contains_any_point + x3_is_in_node_14 <= 0
 R936: - node_14_contains_any_point + x4_is_in_node_14 <= 0
 R937: - node_14_contains_any_point + x5_is_in_node_14 <= 0
 R938: - node_14_contains_any_point + x6_is_in_node_14 <= 0
 R939: - node_14_contains_any_point + x7_is_in_node_14 <= 0
 R940: - node_14_contains_any_point + x8_is_in_node_14 <= 0
 R941: - node_14_contains_any_point + x9_is_in_node_14 <= 0
 R942: - node_14_contains_any_point + x10_is_in_node_14 <= 0
 R943: - node_14_contains_any_point + x11_is_in_node_14 <= 0
 R944: - node_14_contains_any_point + x12_is_in_node_14 <= 0
 R945: - node_14_contains_any_point + x13_is_in_node_14 <= 0
 R946: - node_14_contains_any_point + x14_is_in_node_14 <= 0
 R947: - node_14_contains_any_point + x15_is_in_node_14 <= 0
 R948: - node_14_contains_any_point + x16_is_in_node_14 <= 0
 R949: - node_14_contains_any_point + x17_is_in_node_14 <= 0
 R950: - node_14_contains_any_point + x18_is_in_node_14 <= 0
 R951: - node_14_contains_any_point + x19_is_in_node_14 <= 0
 R952: - node_14_contains_any_point + x20_is_in_node_14 <= 0
 R953: - node_14_contains_any_point + x21_is_in_node_14 <= 0
 R954: - node_14_contains_any_point + x22_is_in_node_14 <= 0
 R955: - node_14_contains_any_point + x23_is_in_node_14 <= 0
 R956: - node_14_contains_any_point + x24_is_in_node_14 <= 0
 R957: - node_14_contains_any_point + x25_is_in_node_14 <= 0
 R958: - node_14_contains_any_point + x26_is_in_node_14 <= 0
 R959: - node_14_contains_any_point + x27_is_in_node_14 <= 0
 R960: - node_14_contains_any_point + x28_is_in_node_14 <= 0
 R961: - node_14_contains_any_point + x29_is_in_node_14 <= 0
 R962: - node_14_contains_any_point + x30_is_in_node_14 <= 0
 R963: - node_14_contains_any_point + x31_is_in_node_14 <= 0
 R964: - node_14_contains_any_point + x32_is_in_node_14 <= 0
 R965: - node_14_contains_any_point + x33_is_in_node_14 <= 0
 R966: - node_14_contains_any_point + x34_is_in_node_14 <= 0
 R967: - node_14_contains_any_point + x35_is_in_node_14 <= 0
 R968: - node_14_contains_any_point + x36_is_in_node_14 <= 0
 R969: - node_14_contains_any_point + x37_is_in_node_14 <= 0
 R970: - node_14_contains_any_point + x38_is_in_node_14 <= 0
 R971: - node_14_contains_any_point + x39_is_in_node_14 <= 0
 R972: - node_14_contains_any_point + x40_is_in_node_14 <= 0
 R973: - node_14_contains_any_point + x41_is_in_node_14 <= 0
 R974: - node_14_contains_any_point + x42_is_in_node_14 <= 0
 R975: - node_14_contains_any_point + x43_is_in_node_14 <= 0
 R976: - node_14_contains_any_point + x44_is_in_node_14 <= 0
 R977: - node_14_contains_any_point + x45_is_in_node_14 <= 0
 R978: - node_14_contains_any_point + x46_is_in_node_14 <= 0
 R979: - node_14_contains_any_point + x47_is_in_node_14 <= 0
 R980: - node_14_contains_any_point + x48_is_in_node_14 <= 0
 R981: - node_14_contains_any_point + x49_is_in_node_14 <= 0
 R982: - node_14_contains_any_point + x50_is_in_node_14 <= 0
 R983: - node_14_contains_any_point + x51_is_in_node_14 <= 0
 R984: - node_14_contains_any_point + x52_is_in_node_14 <= 0
 R985: - node_14_contains_any_point + x53_is_in_node_14 <= 0
 R986: - node_14_contains_any_point + x54_is_in_node_14 <= 0
 R987: - node_14_contains_any_point + x55_is_in_node_14 <= 0
 R988: - node_14_contains_any_point + x56_is_in_node_14 <= 0
 R989: - node_14_contains_any_point + x57_is_in_node_14 <= 0
 R990: - node_14_contains_any_point + x58_is_in_node_14 <= 0
 R991: - node_14_contains_any_point + x59_is_in_node_14 <= 0
 R992: - node_14_contains_any_point + x60_is_in_node_14 <= 0
 R993: - node_14_contains_any_point + x61_is_in_node_14 <= 0
 R994: - node_14_contains_any_point + x62_is_in_node_14 <= 0
 R995: - node_14_contains_any_point + x63_is_in_node_14 <= 0
 R996: - node_14_contains_any_point + x64_is_in_node_14 <= 0
 R997: - node_14_contains_any_point + x65_is_in_node_14 <= 0
 R998: - node_14_contains_any_point + x66_is_in_node_14 <= 0
 R999: - node_14_contains_any_point + x67_is_in_node_14 <= 0
 R1000: - node_14_contains_any_point + x68_is_in_node_14 <= 0
 R1001: - node_14_contains_any_point + x69_is_in_node_14 <= 0
 R1002: - node_14_contains_any_point + x70_is_in_node_14 <= 0
 R1003: - node_14_contains_any_point + x71_is_in_node_14 <= 0
 R1004: - node_14_contains_any_point + x72_is_in_node_14 <= 0
 R1005: - node_14_contains_any_point + x73_is_in_node_14 <= 0
 R1006: - node_14_contains_any_point + x74_is_in_node_14 <= 0
 R1007: - node_14_contains_any_point + x75_is_in_node_14 <= 0
 R1008: - node_14_contains_any_point + x76_is_in_node_14 <= 0
 R1009: - node_14_contains_any_point + x77_is_in_node_14 <= 0
 R1010: - node_14_contains_any_point + x78_is_in_node_14 <= 0
 R1011: - node_14_contains_any_point + x79_is_in_node_14 <= 0
 R1012: - node_14_contains_any_point + x80_is_in_node_14 <= 0
 R1013: - node_14_contains_any_point + x81_is_in_node_14 <= 0
 R1014: - node_14_contains_any_point + x82_is_in_node_14 <= 0
 R1015: - node_14_contains_any_point + x83_is_in_node_14 <= 0
 R1016: - node_14_contains_any_point + x84_is_in_node_14 <= 0
 R1017: - node_14_contains_any_point + x85_is_in_node_14 <= 0
 R1018: - node_14_contains_any_point + x86_is_in_node_14 <= 0
 R1019: - node_14_contains_any_point + x87_is_in_node_14 <= 0
 R1020: - node_14_contains_any_point + x88_is_in_node_14 <= 0
 R1021: - node_14_contains_any_point + x89_is_in_node_14 <= 0
 R1022: - node_14_contains_any_point + x90_is_in_node_14 <= 0
 R1023: - node_14_contains_any_point + x91_is_in_node_14 <= 0
 R1024: - node_14_contains_any_point + x92_is_in_node_14 <= 0
 R1025: - node_14_contains_any_point + x93_is_in_node_14 <= 0
 R1026: - node_14_contains_any_point + x94_is_in_node_14 <= 0
 R1027: - node_14_contains_any_point + x95_is_in_node_14 <= 0
 R1028: - node_14_contains_any_point + x96_is_in_node_14 <= 0
 R1029: - node_14_contains_any_point + x97_is_in_node_14 <= 0
 R1030: - node_14_contains_any_point + x98_is_in_node_14 <= 0
 R1031: - node_14_contains_any_point + x99_is_in_node_14 <= 0
 R1032: - node_14_contains_any_point + x100_is_in_node_14 <= 0
 R1033: - node_14_contains_any_point + x101_is_in_node_14 <= 0
 R1034: - node_14_contains_any_point + x102_is_in_node_14 <= 0
 R1035: - node_14_contains_any_point + x103_is_in_node_14 <= 0
 R1036: - node_14_contains_any_point + x104_is_in_node_14 <= 0
 R1037: - node_14_contains_any_point + x105_is_in_node_14 <= 0
 R1038: - node_14_contains_any_point + x106_is_in_node_14 <= 0
 R1039: - node_14_contains_any_point + x107_is_in_node_14 <= 0
 R1040: - node_14_contains_any_point + x108_is_in_node_14 <= 0
 R1041: - node_14_contains_any_point + x109_is_in_node_14 <= 0
 R1042: - node_14_contains_any_point + x110_is_in_node_14 <= 0
 R1043: - node_14_contains_any_point + x111_is_in_node_14 <= 0
 R1044: - node_14_contains_any_point + x112_is_in_node_14 <= 0
 R1045: - node_14_contains_any_point + x113_is_in_node_14 <= 0
 R1046: - node_14_contains_any_point + x114_is_in_node_14 <= 0
 R1047: - node_14_contains_any_point + x115_is_in_node_14 <= 0
 R1048: - node_14_contains_any_point + x116_is_in_node_14 <= 0
 R1049: - node_14_contains_any_point + x117_is_in_node_14 <= 0
 R1050: - node_14_contains_any_point + x118_is_in_node_14 <= 0
 R1051: - node_14_contains_any_point + x119_is_in_node_14 <= 0
 R1052: - node_14_contains_any_point + x120_is_in_node_14 <= 0
 R1053: - node_14_contains_any_point + x121_is_in_node_14 <= 0
 R1054: - node_14_contains_any_point + x122_is_in_node_14 <= 0
 R1055: - node_14_contains_any_point + x123_is_in_node_14 <= 0
 R1056: - node_14_contains_any_point + x124_is_in_node_14 <= 0
 R1057: - node_14_contains_any_point + x125_is_in_node_14 <= 0
 R1058: - node_14_contains_any_point + x126_is_in_node_14 <= 0
 R1059: - node_14_contains_any_point + x127_is_in_node_14 <= 0
 R1060: - node_14_contains_any_point + x128_is_in_node_14 <= 0
 R1061: - node_14_contains_any_point + x129_is_in_node_14 <= 0
 R1062: - node_14_contains_any_point + x130_is_in_node_14 <= 0
 R1063: - node_14_contains_any_point + x131_is_in_node_14 <= 0
 R1064: - node_14_contains_any_point + x132_is_in_node_14 <= 0
 R1065: - node_14_contains_any_point + x133_is_in_node_14 <= 0
 R1066: - node_14_contains_any_point + x134_is_in_node_14 <= 0
 R1067: - node_14_contains_any_point + x135_is_in_node_14 <= 0
 R1068: - node_14_contains_any_point + x136_is_in_node_14 <= 0
 R1069: - node_14_contains_any_point + x137_is_in_node_14 <= 0
 R1070: - node_14_contains_any_point + x138_is_in_node_14 <= 0
 R1071: - node_14_contains_any_point + x139_is_in_node_14 <= 0
 R1072: - node_14_contains_any_point + x140_is_in_node_14 <= 0
 R1073: - node_14_contains_any_point + x141_is_in_node_14 <= 0
 R1074: - node_14_contains_any_point + x142_is_in_node_14 <= 0
 R1075: - node_14_contains_any_point + x143_is_in_node_14 <= 0
 R1076: - node_14_contains_any_point + x144_is_in_node_14 <= 0
 R1077: - node_14_contains_any_point + x145_is_in_node_14 <= 0
 R1078: - node_14_contains_any_point + x146_is_in_node_14 <= 0
 R1079: - node_14_contains_any_point + x147_is_in_node_14 <= 0
 R1080: - node_14_contains_any_point + x148_is_in_node_14 <= 0
 R1081: - node_14_contains_any_point + x149_is_in_node_14 <= 0
 R1082: - 7 node_14_contains_any_point + x0_is_in_node_14
   + x1_is_in_node_14 + x2_is_in_node_14 + x3_is_in_node_14
   + x4_is_in_node_14 + x5_is_in_node_14 + x6_is_in_node_14
   + x7_is_in_node_14 + x8_is_in_node_14 + x9_is_in_node_14
   + x10_is_in_node_14 + x11_is_in_node_14 + x12_is_in_node_14
   + x13_is_in_node_14 + x14_is_in_node_14 + x15_is_in_node_14
   + x16_is_in_node_14 + x17_is_in_node_14 + x18_is_in_node_14
   + x19_is_in_node_14 + x20_is_in_node_14 + x21_is_in_node_14
   + x22_is_in_node_14 + x23_is_in_node_14 + x24_is_in_node_14
   + x25_is_in_node_14 + x26_is_in_node_14 + x27_is_in_node_14
   + x28_is_in_node_14 + x29_is_in_node_14 + x30_is_in_node_14
   + x31_is_in_node_14 + x32_is_in_node_14 + x33_is_in_node_14
   + x34_is_in_node_14 + x35_is_in_node_14 + x36_is_in_node_14
   + x37_is_in_node_14 + x38_is_in_node_14 + x39_is_in_node_14
   + x40_is_in_node_14 + x41_is_in_node_14 + x42_is_in_node_14
   + x43_is_in_node_14 + x44_is_in_node_14 + x45_is_in_node_14
   + x46_is_in_node_14 + x47_is_in_node_14 + x48_is_in_node_14
   + x49_is_in_node_14 + x50_is_in_node_14 + x51_is_in_node_14
   + x52_is_in_node_14 + x53_is_in_node_14 + x54_is_in_node_14
   + x55_is_in_node_14 + x56_is_in_node_14 + x57_is_in_node_14
   + x58_is_in_node_14 + x59_is_in_node_14 + x60_is_in_node_14
   + x61_is_in_node_14 + x62_is_in_node_14 + x63_is_in_node_14
   + x64_is_in_node_14 + x65_is_in_node_14 + x66_is_in_node_14
   + x67_is_in_node_14 + x68_is_in_node_14 + x69_is_in_node_14
   + x70_is_in_node_14 + x71_is_in_node_14 + x72_is_in_node_14
   + x73_is_in_node_14 + x74_is_in_node_14 + x75_is_in_node_14
   + x76_is_in_node_14 + x77_is_in_node_14 + x78_is_in_node_14
   + x79_is_in_node_14 + x80_is_in_node_14 + x81_is_in_node_14
   + x82_is_in_node_14 + x83_is_in_node_14 + x84_is_in_node_14
   + x85_is_in_node_14 + x86_is_in_node_14 + x87_is_in_node_14
   + x88_is_in_node_14 + x89_is_in_node_14 + x90_is_in_node_14
   + x91_is_in_node_14 + x92_is_in_node_14 + x93_is_in_node_14
   + x94_is_in_node_14 + x95_is_in_node_14 + x96_is_in_node_14
   + x97_is_in_node_14 + x98_is_in_node_14 + x99_is_in_node_14
   + x100_is_in_node_14 + x101_is_in_node_14 + x102_is_in_node_14
   + x103_is_in_node_14 + x104_is_in_node_14 + x105_is_in_node_14
   + x106_is_in_node_14 + x107_is_in_node_14 + x108_is_in_node_14
   + x109_is_in_node_14 + x110_is_in_node_14 + x111_is_in_node_14
   + x112_is_in_node_14 + x113_is_in_node_14 + x114_is_in_node_14
   + x115_is_in_node_14 + x116_is_in_node_14 + x117_is_in_node_14
   + x118_is_in_node_14 + x119_is_in_node_14 + x120_is_in_node_14
   + x121_is_in_node_14 + x122_is_in_node_14 + x123_is_in_node_14
   + x124_is_in_node_14 + x125_is_in_node_14 + x126_is_in_node_14
   + x127_is_in_node_14 + x128_is_in_node_14 + x129_is_in_node_14
   + x130_is_in_node_14 + x131_is_in_node_14 + x132_is_in_node_14
   + x133_is_in_node_14 + x134_is_in_node_14 + x135_is_in_node_14
   + x136_is_in_node_14 + x137_is_in_node_14 + x138_is_in_node_14
   + x139_is_in_node_14 + x140_is_in_node_14 + x141_is_in_node_14
   + x142_is_in_node_14 + x143_is_in_node_14 + x144_is_in_node_14
   + x145_is_in_node_14 + x146_is_in_node_14 + x147_is_in_node_14
   + x148_is_in_node_14 + x149_is_in_node_14 >= 0
 R1083: x0_is_in_node_14 + x1_is_in_node_14 + x2_is_in_node_14
   + x3_is_in_node_14 + x4_is_in_node_14 + x5_is_in_node_14
   + x6_is_in_node_14 + x7_is_in_node_14 + x8_is_in_node_14
   + x9_is_in_node_14 + x10_is_in_node_14 + x11_is_in_node_14
   + x12_is_in_node_14 + x13_is_in_node_14 + x14_is_in_node_14
   + x15_is_in_node_14 + x16_is_in_node_14 + x17_is_in_node_14
   + x18_is_in_node_14 + x19_is_in_node_14 + x20_is_in_node_14
   + x21_is_in_node_14 + x22_is_in_node_14 + x23_is_in_node_14
   + x24_is_in_node_14 + x25_is_in_node_14 + x26_is_in_node_14
   + x27_is_in_node_14 + x28_is_in_node_14 + x29_is_in_node_14
   + x30_is_in_node_14 + x31_is_in_node_14 + x32_is_in_node_14
   + x33_is_in_node_14 + x34_is_in_node_14 + x35_is_in_node_14
   + x36_is_in_node_14 + x37_is_in_node_14 + x38_is_in_node_14
   + x39_is_in_node_14 + x40_is_in_node_14 + x41_is_in_node_14
   + x42_is_in_node_14 + x43_is_in_node_14 + x44_is_in_node_14
   + x45_is_in_node_14 + x46_is_in_node_14 + x47_is_in_node_14
   + x48_is_in_node_14 + x49_is_in_node_14 + x50_is_in_node_14
   + x51_is_in_node_14 + x52_is_in_node_14 + x53_is_in_node_14
   + x54_is_in_node_14 + x55_is_in_node_14 + x56_is_in_node_14
   + x57_is_in_node_14 + x58_is_in_node_14 + x59_is_in_node_14
   + x60_is_in_node_14 + x61_is_in_node_14 + x62_is_in_node_14
   + x63_is_in_node_14 + x64_is_in_node_14 + x65_is_in_node_14
   + x66_is_in_node_14 + x67_is_in_node_14 + x68_is_in_node_14
   + x69_is_in_node_14 + x70_is_in_node_14 + x71_is_in_node_14
   + x72_is_in_node_14 + x73_is_in_node_14 + x74_is_in_node_14
   + x75_is_in_node_14 + x76_is_in_node_14 + x77_is_in_node_14
   + x78_is_in_node_14 + x79_is_in_node_14 + x80_is_in_node_14
   + x81_is_in_node_14 + x82_is_in_node_14 + x83_is_in_node_14
   + x84_is_in_node_14 + x85_is_in_node_14 + x86_is_in_node_14
   + x87_is_in_node_14 + x88_is_in_node_14 + x89_is_in_node_14
   + x90_is_in_node_14 + x91_is_in_node_14 + x92_is_in_node_14
   + x93_is_in_node_14 + x94_is_in_node_14 + x95_is_in_node_14
   + x96_is_in_node_14 + x97_is_in_node_14 + x98_is_in_node_14
   + x99_is_in_node_14 + x100_is_in_node_14 + x101_is_in_node_14
   + x102_is_in_node_14 + x103_is_in_node_14 + x104_is_in_node_14
   + x105_is_in_node_14 + x106_is_in_node_14 + x107_is_in_node_14
   + x108_is_in_node_14 + x109_is_in_node_14 + x110_is_in_node_14
   + x111_is_in_node_14 + x112_is_in_node_14 + x113_is_in_node_14
   + x114_is_in_node_14 + x115_is_in_node_14 + x116_is_in_node_14
   + x117_is_in_node_14 + x118_is_in_node_14 + x119_is_in_node_14
   + x120_is_in_node_14 + x121_is_in_node_14 + x122_is_in_node_14
   + x123_is_in_node_14 + x124_is_in_node_14 + x125_is_in_node_14
   + x126_is_in_node_14 + x127_is_in_node_14 + x128_is_in_node_14
   + x129_is_in_node_14 + x130_is_in_node_14 + x131_is_in_node_14
   + x132_is_in_node_14 + x133_is_in_node_14 + x134_is_in_node_14
   + x135_is_in_node_14 + x136_is_in_node_14 + x137_is_in_node_14
   + x138_is_in_node_14 + x139_is_in_node_14 + x140_is_in_node_14
   + x141_is_in_node_14 + x142_is_in_node_14 + x143_is_in_node_14
   + x144_is_in_node_14 + x145_is_in_node_14 + x146_is_in_node_14
   + x147_is_in_node_14 + x148_is_in_node_14 + x149_is_in_node_14 = 1
 R1084: - node_15_contains_any_point + x0_is_in_node_15 <= 0
 R1085: - node_15_contains_any_point + x1_is_in_node_15 <= 0
 R1086: - node_15_contains_any_point + x2_is_in_node_15 <= 0
 R1087: - node_15_contains_any_point + x3_is_in_node_15 <= 0
 R1088: - node_15_contains_any_point + x4_is_in_node_15 <= 0
 R1089: - node_15_contains_any_point + x5_is_in_node_15 <= 0
 R1090: - node_15_contains_any_point + x6_is_in_node_15 <= 0
 R1091: - node_15_contains_any_point + x7_is_in_node_15 <= 0
 R1092: - node_15_contains_any_point + x8_is_in_node_15 <= 0
 R1093: - node_15_contains_any_point + x9_is_in_node_15 <= 0
 R1094: - node_15_contains_any_point + x10_is_in_node_15 <= 0
 R1095: - node_15_contains_any_point + x11_is_in_node_15 <= 0
 R1096: - node_15_contains_any_point + x12_is_in_node_15 <= 0
 R1097: - node_15_contains_any_point + x13_is_in_node_15 <= 0
 R1098: - node_15_contains_any_point + x14_is_in_node_15 <= 0
 R1099: - node_15_contains_any_point + x15_is_in_node_15 <= 0
 R1100: - node_15_contains_any_point + x16_is_in_node_15 <= 0
 R1101: - node_15_contains_any_point + x17_is_in_node_15 <= 0
 R1102: - node_15_contains_any_point + x18_is_in_node_15 <= 0
 R1103: - node_15_contains_any_point + x19_is_in_node_15 <= 0
 R1104: - node_15_contains_any_point + x20_is_in_node_15 <= 0
 R1105: - node_15_contains_any_point + x21_is_in_node_15 <= 0
 R1106: - node_15_contains_any_point + x22_is_in_node_15 <= 0
 R1107: - node_15_contains_any_point + x23_is_in_node_15 <= 0
 R1108: - node_15_contains_any_point + x24_is_in_node_15 <= 0
 R1109: - node_15_contains_any_point + x25_is_in_node_15 <= 0
 R1110: - node_15_contains_any_point + x26_is_in_node_15 <= 0
 R1111: - node_15_contains_any_point + x27_is_in_node_15 <= 0
 R1112: - node_15_contains_any_point + x28_is_in_node_15 <= 0
 R1113: - node_15_contains_any_point + x29_is_in_node_15 <= 0
 R1114: - node_15_contains_any_point + x30_is_in_node_15 <= 0
 R1115: - node_15_contains_any_point + x31_is_in_node_15 <= 0
 R1116: - node_15_contains_any_point + x32_is_in_node_15 <= 0
 R1117: - node_15_contains_any_point + x33_is_in_node_15 <= 0
 R1118: - node_15_contains_any_point + x34_is_in_node_15 <= 0
 R1119: - node_15_contains_any_point + x35_is_in_node_15 <= 0
 R1120: - node_15_contains_any_point + x36_is_in_node_15 <= 0
 R1121: - node_15_contains_any_point + x37_is_in_node_15 <= 0
 R1122: - node_15_contains_any_point + x38_is_in_node_15 <= 0
 R1123: - node_15_contains_any_point + x39_is_in_node_15 <= 0
 R1124: - node_15_contains_any_point + x40_is_in_node_15 <= 0
 R1125: - node_15_contains_any_point + x41_is_in_node_15 <= 0
 R1126: - node_15_contains_any_point + x42_is_in_node_15 <= 0
 R1127: - node_15_contains_any_point + x43_is_in_node_15 <= 0
 R1128: - node_15_contains_any_point + x44_is_in_node_15 <= 0
 R1129: - node_15_contains_any_point + x45_is_in_node_15 <= 0
 R1130: - node_15_contains_any_point + x46_is_in_node_15 <= 0
 R1131: - node_15_contains_any_point + x47_is_in_node_15 <= 0
 R1132: - node_15_contains_any_point + x48_is_in_node_15 <= 0
 R1133: - node_15_contains_any_point + x49_is_in_node_15 <= 0
 R1134: - node_15_contains_any_point + x50_is_in_node_15 <= 0
 R1135: - node_15_contains_any_point + x51_is_in_node_15 <= 0
 R1136: - node_15_contains_any_point + x52_is_in_node_15 <= 0
 R1137: - node_15_contains_any_point + x53_is_in_node_15 <= 0
 R1138: - node_15_contains_any_point + x54_is_in_node_15 <= 0
 R1139: - node_15_contains_any_point + x55_is_in_node_15 <= 0
 R1140: - node_15_contains_any_point + x56_is_in_node_15 <= 0
 R1141: - node_15_contains_any_point + x57_is_in_node_15 <= 0
 R1142: - node_15_contains_any_point + x58_is_in_node_15 <= 0
 R1143: - node_15_contains_any_point + x59_is_in_node_15 <= 0
 R1144: - node_15_contains_any_point + x60_is_in_node_15 <= 0
 R1145: - node_15_contains_any_point + x61_is_in_node_15 <= 0
 R1146: - node_15_contains_any_point + x62_is_in_node_15 <= 0
 R1147: - node_15_contains_any_point + x63_is_in_node_15 <= 0
 R1148: - node_15_contains_any_point + x64_is_in_node_15 <= 0
 R1149: - node_15_contains_any_point + x65_is_in_node_15 <= 0
 R1150: - node_15_contains_any_point + x66_is_in_node_15 <= 0
 R1151: - node_15_contains_any_point + x67_is_in_node_15 <= 0
 R1152: - node_15_contains_any_point + x68_is_in_node_15 <= 0
 R1153: - node_15_contains_any_point + x69_is_in_node_15 <= 0
 R1154: - node_15_contains_any_point + x70_is_in_node_15 <= 0
 R1155: - node_15_contains_any_point + x71_is_in_node_15 <= 0
 R1156: - node_15_contains_any_point + x72_is_in_node_15 <= 0
 R1157: - node_15_contains_any_point + x73_is_in_node_15 <= 0
 R1158: - node_15_contains_any_point + x74_is_in_node_15 <= 0
 R1159: - node_15_contains_any_point + x75_is_in_node_15 <= 0
 R1160: - node_15_contains_any_point + x76_is_in_node_15 <= 0
 R1161: - node_15_contains_any_point + x77_is_in_node_15 <= 0
 R1162: - node_15_contains_any_point + x78_is_in_node_15 <= 0
 R1163: - node_15_contains_any_point + x79_is_in_node_15 <= 0
 R1164: - node_15_contains_any_point + x80_is_in_node_15 <= 0
 R1165: - node_15_contains_any_point + x81_is_in_node_15 <= 0
 R1166: - node_15_contains_any_point + x82_is_in_node_15 <= 0
 R1167: - node_15_contains_any_point + x83_is_in_node_15 <= 0
 R1168: - node_15_contains_any_point + x84_is_in_node_15 <= 0
 R1169: - node_15_contains_any_point + x85_is_in_node_15 <= 0
 R1170: - node_15_contains_any_point + x86_is_in_node_15 <= 0
 R1171: - node_15_contains_any_point + x87_is_in_node_15 <= 0
 R1172: - node_15_contains_any_point + x88_is_in_node_15 <= 0
 R1173: - node_15_contains_any_point + x89_is_in_node_15 <= 0
 R1174: - node_15_contains_any_point + x90_is_in_node_15 <= 0
 R1175: - node_15_contains_any_point + x91_is_in_node_15 <= 0
 R1176: - node_15_contains_any_point + x92_is_in_node_15 <= 0
 R1177: - node_15_contains_any_point + x93_is_in_node_15 <= 0
 R1178: - node_15_contains_any_point + x94_is_in_node_15 <= 0
 R1179: - node_15_contains_any_point + x95_is_in_node_15 <= 0
 R1180: - node_15_contains_any_point + x96_is_in_node_15 <= 0
 R1181: - node_15_contains_any_point + x97_is_in_node_15 <= 0
 R1182: - node_15_contains_any_point + x98_is_in_node_15 <= 0
 R1183: - node_15_contains_any_point + x99_is_in_node_15 <= 0
 R1184: - node_15_contains_any_point + x100_is_in_node_15 <= 0
 R1185: - node_15_contains_any_point + x101_is_in_node_15 <= 0
 R1186: - node_15_contains_any_point + x102_is_in_node_15 <= 0
 R1187: - node_15_contains_any_point + x103_is_in_node_15 <= 0
 R1188: - node_15_contains_any_point + x104_is_in_node_15 <= 0
 R1189: - node_15_contains_any_point + x105_is_in_node_15 <= 0
 R1190: - node_15_contains_any_point + x106_is_in_node_15 <= 0
 R1191: - node_15_contains_any_point + x107_is_in_node_15 <= 0
 R1192: - node_15_contains_any_point + x108_is_in_node_15 <= 0
 R1193: - node_15_contains_any_point + x109_is_in_node_15 <= 0
 R1194: - node_15_contains_any_point + x110_is_in_node_15 <= 0
 R1195: - node_15_contains_any_point + x111_is_in_node_15 <= 0
 R1196: - node_15_contains_any_point + x112_is_in_node_15 <= 0
 R1197: - node_15_contains_any_point + x113_is_in_node_15 <= 0
 R1198: - node_15_contains_any_point + x114_is_in_node_15 <= 0
 R1199: - node_15_contains_any_point + x115_is_in_node_15 <= 0
 R1200: - node_15_contains_any_point + x116_is_in_node_15 <= 0
 R1201: - node_15_contains_any_point + x117_is_in_node_15 <= 0
 R1202: - node_15_contains_any_point + x118_is_in_node_15 <= 0
 R1203: - node_15_contains_any_point + x119_is_in_node_15 <= 0
 R1204: - node_15_contains_any_point + x120_is_in_node_15 <= 0
 R1205: - node_15_contains_any_point + x121_is_in_node_15 <= 0
 R1206: - node_15_contains_any_point + x122_is_in_node_15 <= 0
 R1207: - node_15_contains_any_point + x123_is_in_node_15 <= 0
 R1208: - node_15_contains_any_point + x124_is_in_node_15 <= 0
 R1209: - node_15_contains_any_point + x125_is_in_node_15 <= 0
 R1210: - node_15_contains_any_point + x126_is_in_node_15 <= 0
 R1211: - node_15_contains_any_point + x127_is_in_node_15 <= 0
 R1212: - node_15_contains_any_point + x128_is_in_node_15 <= 0
 R1213: - node_15_contains_any_point + x129_is_in_node_15 <= 0
 R1214: - node_15_contains_any_point + x130_is_in_node_15 <= 0
 R1215: - node_15_contains_any_point + x131_is_in_node_15 <= 0
 R1216: - node_15_contains_any_point + x132_is_in_node_15 <= 0
 R1217: - node_15_contains_any_point + x133_is_in_node_15 <= 0
 R1218: - node_15_contains_any_point + x134_is_in_node_15 <= 0
 R1219: - node_15_contains_any_point + x135_is_in_node_15 <= 0
 R1220: - node_15_contains_any_point + x136_is_in_node_15 <= 0
 R1221: - node_15_contains_any_point + x137_is_in_node_15 <= 0
 R1222: - node_15_contains_any_point + x138_is_in_node_15 <= 0
 R1223: - node_15_contains_any_point + x139_is_in_node_15 <= 0
 R1224: - node_15_contains_any_point + x140_is_in_node_15 <= 0
 R1225: - node_15_contains_any_point + x141_is_in_node_15 <= 0
 R1226: - node_15_contains_any_point + x142_is_in_node_15 <= 0
 R1227: - node_15_contains_any_point + x143_is_in_node_15 <= 0
 R1228: - node_15_contains_any_point + x144_is_in_node_15 <= 0
 R1229: - node_15_contains_any_point + x145_is_in_node_15 <= 0
 R1230: - node_15_contains_any_point + x146_is_in_node_15 <= 0
 R1231: - node_15_contains_any_point + x147_is_in_node_15 <= 0
 R1232: - node_15_contains_any_point + x148_is_in_node_15 <= 0
 R1233: - node_15_contains_any_point + x149_is_in_node_15 <= 0
 R1234: - 7 node_15_contains_any_point + x0_is_in_node_15
   + x1_is_in_node_15 + x2_is_in_node_15 + x3_is_in_node_15
   + x4_is_in_node_15 + x5_is_in_node_15 + x6_is_in_node_15
   + x7_is_in_node_15 + x8_is_in_node_15 + x9_is_in_node_15
   + x10_is_in_node_15 + x11_is_in_node_15 + x12_is_in_node_15
   + x13_is_in_node_15 + x14_is_in_node_15 + x15_is_in_node_15
   + x16_is_in_node_15 + x17_is_in_node_15 + x18_is_in_node_15
   + x19_is_in_node_15 + x20_is_in_node_15 + x21_is_in_node_15
   + x22_is_in_node_15 + x23_is_in_node_15 + x24_is_in_node_15
   + x25_is_in_node_15 + x26_is_in_node_15 + x27_is_in_node_15
   + x28_is_in_node_15 + x29_is_in_node_15 + x30_is_in_node_15
   + x31_is_in_node_15 + x32_is_in_node_15 + x33_is_in_node_15
   + x34_is_in_node_15 + x35_is_in_node_15 + x36_is_in_node_15
   + x37_is_in_node_15 + x38_is_in_node_15 + x39_is_in_node_15
   + x40_is_in_node_15 + x41_is_in_node_15 + x42_is_in_node_15
   + x43_is_in_node_15 + x44_is_in_node_15 + x45_is_in_node_15
   + x46_is_in_node_15 + x47_is_in_node_15 + x48_is_in_node_15
   + x49_is_in_node_15 + x50_is_in_node_15 + x51_is_in_node_15
   + x52_is_in_node_15 + x53_is_in_node_15 + x54_is_in_node_15
   + x55_is_in_node_15 + x56_is_in_node_15 + x57_is_in_node_15
   + x58_is_in_node_15 + x59_is_in_node_15 + x60_is_in_node_15
   + x61_is_in_node_15 + x62_is_in_node_15 + x63_is_in_node_15
   + x64_is_in_node_15 + x65_is_in_node_15 + x66_is_in_node_15
   + x67_is_in_node_15 + x68_is_in_node_15 + x69_is_in_node_15
   + x70_is_in_node_15 + x71_is_in_node_15 + x72_is_in_node_15
   + x73_is_in_node_15 + x74_is_in_node_15 + x75_is_in_node_15
   + x76_is_in_node_15 + x77_is_in_node_15 + x78_is_in_node_15
   + x79_is_in_node_15 + x80_is_in_node_15 + x81_is_in_node_15
   + x82_is_in_node_15 + x83_is_in_node_15 + x84_is_in_node_15
   + x85_is_in_node_15 + x86_is_in_node_15 + x87_is_in_node_15
   + x88_is_in_node_15 + x89_is_in_node_15 + x90_is_in_node_15
   + x91_is_in_node_15 + x92_is_in_node_15 + x93_is_in_node_15
   + x94_is_in_node_15 + x95_is_in_node_15 + x96_is_in_node_15
   + x97_is_in_node_15 + x98_is_in_node_15 + x99_is_in_node_15
   + x100_is_in_node_15 + x101_is_in_node_15 + x102_is_in_node_15
   + x103_is_in_node_15 + x104_is_in_node_15 + x105_is_in_node_15
   + x106_is_in_node_15 + x107_is_in_node_15 + x108_is_in_node_15
   + x109_is_in_node_15 + x110_is_in_node_15 + x111_is_in_node_15
   + x112_is_in_node_15 + x113_is_in_node_15 + x114_is_in_node_15
   + x115_is_in_node_15 + x116_is_in_node_15 + x117_is_in_node_15
   + x118_is_in_node_15 + x119_is_in_node_15 + x120_is_in_node_15
   + x121_is_in_node_15 + x122_is_in_node_15 + x123_is_in_node_15
   + x124_is_in_node_15 + x125_is_in_node_15 + x126_is_in_node_15
   + x127_is_in_node_15 + x128_is_in_node_15 + x129_is_in_node_15
   + x130_is_in_node_15 + x131_is_in_node_15 + x132_is_in_node_15
   + x133_is_in_node_15 + x134_is_in_node_15 + x135_is_in_node_15
   + x136_is_in_node_15 + x137_is_in_node_15 + x138_is_in_node_15
   + x139_is_in_node_15 + x140_is_in_node_15 + x141_is_in_node_15
   + x142_is_in_node_15 + x143_is_in_node_15 + x144_is_in_node_15
   + x145_is_in_node_15 + x146_is_in_node_15 + x147_is_in_node_15
   + x148_is_in_node_15 + x149_is_in_node_15 >= 0
 R1235: x0_is_in_node_15 + x1_is_in_node_15 + x2_is_in_node_15
   + x3_is_in_node_15 + x4_is_in_node_15 + x5_is_in_node_15
   + x6_is_in_node_15 + x7_is_in_node_15 + x8_is_in_node_15
   + x9_is_in_node_15 + x10_is_in_node_15 + x11_is_in_node_15
   + x12_is_in_node_15 + x13_is_in_node_15 + x14_is_in_node_15
   + x15_is_in_node_15 + x16_is_in_node_15 + x17_is_in_node_15
   + x18_is_in_node_15 + x19_is_in_node_15 + x20_is_in_node_15
   + x21_is_in_node_15 + x22_is_in_node_15 + x23_is_in_node_15
   + x24_is_in_node_15 + x25_is_in_node_15 + x26_is_in_node_15
   + x27_is_in_node_15 + x28_is_in_node_15 + x29_is_in_node_15
   + x30_is_in_node_15 + x31_is_in_node_15 + x32_is_in_node_15
   + x33_is_in_node_15 + x34_is_in_node_15 + x35_is_in_node_15
   + x36_is_in_node_15 + x37_is_in_node_15 + x38_is_in_node_15
   + x39_is_in_node_15 + x40_is_in_node_15 + x41_is_in_node_15
   + x42_is_in_node_15 + x43_is_in_node_15 + x44_is_in_node_15
   + x45_is_in_node_15 + x46_is_in_node_15 + x47_is_in_node_15
   + x48_is_in_node_15 + x49_is_in_node_15 + x50_is_in_node_15
   + x51_is_in_node_15 + x52_is_in_node_15 + x53_is_in_node_15
   + x54_is_in_node_15 + x55_is_in_node_15 + x56_is_in_node_15
   + x57_is_in_node_15 + x58_is_in_node_15 + x59_is_in_node_15
   + x60_is_in_node_15 + x61_is_in_node_15 + x62_is_in_node_15
   + x63_is_in_node_15 + x64_is_in_node_15 + x65_is_in_node_15
   + x66_is_in_node_15 + x67_is_in_node_15 + x68_is_in_node_15
   + x69_is_in_node_15 + x70_is_in_node_15 + x71_is_in_node_15
   + x72_is_in_node_15 + x73_is_in_node_15 + x74_is_in_node_15
   + x75_is_in_node_15 + x76_is_in_node_15 + x77_is_in_node_15
   + x78_is_in_node_15 + x79_is_in_node_15 + x80_is_in_node_15
   + x81_is_in_node_15 + x82_is_in_node_15 + x83_is_in_node_15
   + x84_is_in_node_15 + x85_is_in_node_15 + x86_is_in_node_15
   + x87_is_in_node_15 + x88_is_in_node_15 + x89_is_in_node_15
   + x90_is_in_node_15 + x91_is_in_node_15 + x92_is_in_node_15
   + x93_is_in_node_15 + x94_is_in_node_15 + x95_is_in_node_15
   + x96_is_in_node_15 + x97_is_in_node_15 + x98_is_in_node_15
   + x99_is_in_node_15 + x100_is_in_node_15 + x101_is_in_node_15
   + x102_is_in_node_15 + x103_is_in_node_15 + x104_is_in_node_15
   + x105_is_in_node_15 + x106_is_in_node_15 + x107_is_in_node_15
   + x108_is_in_node_15 + x109_is_in_node_15 + x110_is_in_node_15
   + x111_is_in_node_15 + x112_is_in_node_15 + x113_is_in_node_15
   + x114_is_in_node_15 + x115_is_in_node_15 + x116_is_in_node_15
   + x117_is_in_node_15 + x118_is_in_node_15 + x119_is_in_node_15
   + x120_is_in_node_15 + x121_is_in_node_15 + x122_is_in_node_15
   + x123_is_in_node_15 + x124_is_in_node_15 + x125_is_in_node_15
   + x126_is_in_node_15 + x127_is_in_node_15 + x128_is_in_node_15
   + x129_is_in_node_15 + x130_is_in_node_15 + x131_is_in_node_15
   + x132_is_in_node_15 + x133_is_in_node_15 + x134_is_in_node_15
   + x135_is_in_node_15 + x136_is_in_node_15 + x137_is_in_node_15
   + x138_is_in_node_15 + x139_is_in_node_15 + x140_is_in_node_15
   + x141_is_in_node_15 + x142_is_in_node_15 + x143_is_in_node_15
   + x144_is_in_node_15 + x145_is_in_node_15 + x146_is_in_node_15
   + x147_is_in_node_15 + x148_is_in_node_15 + x149_is_in_node_15 = 1
 R1236: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_8 <= 1.1
 R1237: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x0_is_in_node_8 >= -1
 R1238: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x0_is_in_node_8 >= -1
 R1239: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_8 <= 1.1
 R1240: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x1_is_in_node_8 >= -1
 R1241: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x1_is_in_node_8 >= -1
 R1242: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_8 <= 1.1
 R1243: - split_value_node_4 + 4.7 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x2_is_in_node_8 >= -1
 R1244: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x2_is_in_node_8 >= -1
 R1245: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_8 <= 1.1
 R1246: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x3_is_in_node_8 >= -1
 R1247: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x3_is_in_node_8 >= -1
 R1248: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_8 <= 1.1
 R1249: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.6 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x4_is_in_node_8 >= -1
 R1250: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x4_is_in_node_8 >= -1
 R1251: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_8 <= 1.1
 R1252: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.9 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x5_is_in_node_8 >= -1
 R1253: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x5_is_in_node_8 >= -1
 R1254: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_8 <= 1.1
 R1255: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x6_is_in_node_8 >= -1
 R1256: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x6_is_in_node_8 >= -1
 R1257: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_8 <= 1.1
 R1258: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x7_is_in_node_8 >= -1
 R1259: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x7_is_in_node_8 >= -1
 R1260: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_8 <= 1.1
 R1261: - split_value_node_4 + 4.4 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x8_is_in_node_8 >= -1
 R1262: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x8_is_in_node_8 >= -1
 R1263: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_8 <= 1.1
 R1264: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x9_is_in_node_8 >= -1
 R1265: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x9_is_in_node_8 >= -1
 R1266: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_8 <= 1.1
 R1267: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.7 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x10_is_in_node_8 >= -1
 R1268: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x10_is_in_node_8 >= -1
 R1269: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_8 <= 1.1
 R1270: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x11_is_in_node_8 >= -1
 R1271: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x11_is_in_node_8 >= -1
 R1272: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_8 <= 1.1
 R1273: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x12_is_in_node_8 >= -1
 R1274: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x12_is_in_node_8 >= -1
 R1275: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_8 <= 1.1
 R1276: - split_value_node_4 + 4.3 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.1 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x13_is_in_node_8 >= -1
 R1277: - split_value_node_2 + 4.3 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.1 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x13_is_in_node_8 >= -1
 R1278: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_8 <= 1.1
 R1279: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 4 node_4_considers_var_1 + 1.2 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x14_is_in_node_8 >= -1
 R1280: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 4 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x14_is_in_node_8 >= -1
 R1281: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_8 <= 1.1
 R1282: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 4.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x15_is_in_node_8 >= -1
 R1283: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 4.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x15_is_in_node_8 >= -1
 R1284: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_8 <= 1.1
 R1285: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.9 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x16_is_in_node_8 >= -1
 R1286: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x16_is_in_node_8 >= -1
 R1287: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_8 <= 1.1
 R1288: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x17_is_in_node_8 >= -1
 R1289: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x17_is_in_node_8 >= -1
 R1290: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_8 <= 1.1
 R1291: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x18_is_in_node_8 >= -1
 R1292: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x18_is_in_node_8 >= -1
 R1293: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_8 <= 1.1
 R1294: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x19_is_in_node_8 >= -1
 R1295: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x19_is_in_node_8 >= -1
 R1296: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_8 <= 1.1
 R1297: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x20_is_in_node_8 >= -1
 R1298: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x20_is_in_node_8 >= -1
 R1299: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_8 <= 1.1
 R1300: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.7 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x21_is_in_node_8 >= -1
 R1301: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x21_is_in_node_8 >= -1
 R1302: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_8 <= 1.1
 R1303: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.6 node_4_considers_var_1 + node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x22_is_in_node_8 >= -1
 R1304: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x22_is_in_node_8 >= -1
 R1305: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_8 <= 1.1
 R1306: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.5 node_4_considers_var_3 - x23_is_in_node_8 >= -1
 R1307: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.5 node_2_considers_var_3 - x23_is_in_node_8 >= -1
 R1308: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_8 <= 1.1
 R1309: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.9 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x24_is_in_node_8 >= -1
 R1310: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x24_is_in_node_8 >= -1
 R1311: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_8 <= 1.1
 R1312: - split_value_node_4 + 5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x25_is_in_node_8 >= -1
 R1313: - split_value_node_2 + 5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x25_is_in_node_8 >= -1
 R1314: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_8 <= 1.1
 R1315: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x26_is_in_node_8 >= -1
 R1316: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x26_is_in_node_8 >= -1
 R1317: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_8 <= 1.1
 R1318: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x27_is_in_node_8 >= -1
 R1319: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x27_is_in_node_8 >= -1
 R1320: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_8 <= 1.1
 R1321: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x28_is_in_node_8 >= -1
 R1322: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x28_is_in_node_8 >= -1
 R1323: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_8 <= 1.1
 R1324: - split_value_node_4 + 4.7 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x29_is_in_node_8 >= -1
 R1325: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x29_is_in_node_8 >= -1
 R1326: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_8 <= 1.1
 R1327: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x30_is_in_node_8 >= -1
 R1328: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x30_is_in_node_8 >= -1
 R1329: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_8 <= 1.1
 R1330: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x31_is_in_node_8 >= -1
 R1331: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x31_is_in_node_8 >= -1
 R1332: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_8 <= 1.1
 R1333: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 4.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x32_is_in_node_8 >= -1
 R1334: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 4.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x32_is_in_node_8 >= -1
 R1335: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_8 <= 1.1
 R1336: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 4.2 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x33_is_in_node_8 >= -1
 R1337: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 4.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x33_is_in_node_8 >= -1
 R1338: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_8 <= 1.1
 R1339: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x34_is_in_node_8 >= -1
 R1340: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x34_is_in_node_8 >= -1
 R1341: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_8 <= 1.1
 R1342: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.2 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x35_is_in_node_8 >= -1
 R1343: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x35_is_in_node_8 >= -1
 R1344: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_8 <= 1.1
 R1345: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x36_is_in_node_8 >= -1
 R1346: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x36_is_in_node_8 >= -1
 R1347: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_8 <= 1.1
 R1348: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x37_is_in_node_8 >= -1
 R1349: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x37_is_in_node_8 >= -1
 R1350: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_8 <= 1.1
 R1351: - split_value_node_4 + 4.4 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x38_is_in_node_8 >= -1
 R1352: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x38_is_in_node_8 >= -1
 R1353: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_8 <= 1.1
 R1354: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x39_is_in_node_8 >= -1
 R1355: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x39_is_in_node_8 >= -1
 R1356: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_8 <= 1.1
 R1357: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x40_is_in_node_8 >= -1
 R1358: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x40_is_in_node_8 >= -1
 R1359: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_8 <= 1.1
 R1360: - split_value_node_4 + 4.5 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x41_is_in_node_8 >= -1
 R1361: - split_value_node_2 + 4.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x41_is_in_node_8 >= -1
 R1362: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_8 <= 1.1
 R1363: - split_value_node_4 + 4.4 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x42_is_in_node_8 >= -1
 R1364: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x42_is_in_node_8 >= -1
 R1365: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_8 <= 1.1
 R1366: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.6 node_4_considers_var_3 - x43_is_in_node_8 >= -1
 R1367: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.6 node_2_considers_var_3 - x43_is_in_node_8 >= -1
 R1368: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_8 <= 1.1
 R1369: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.9 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x44_is_in_node_8 >= -1
 R1370: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x44_is_in_node_8 >= -1
 R1371: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_8 <= 1.1
 R1372: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x45_is_in_node_8 >= -1
 R1373: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x45_is_in_node_8 >= -1
 R1374: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_8 <= 1.1
 R1375: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x46_is_in_node_8 >= -1
 R1376: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x46_is_in_node_8 >= -1
 R1377: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_8 <= 1.1
 R1378: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x47_is_in_node_8 >= -1
 R1379: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x47_is_in_node_8 >= -1
 R1380: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_8 <= 1.1
 R1381: - split_value_node_4 + 5.3 node_4_considers_var_0
   + 3.7 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x48_is_in_node_8 >= -1
 R1382: - split_value_node_2 + 5.3 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x48_is_in_node_8 >= -1
 R1383: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_8 <= 1.1
 R1384: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x49_is_in_node_8 >= -1
 R1385: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x49_is_in_node_8 >= -1
 R1386: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_8 <= 1.1
 R1387: - split_value_node_4 + 7 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x50_is_in_node_8 >= -1
 R1388: - split_value_node_2 + 7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x50_is_in_node_8 >= -1
 R1389: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_8 <= 1.1
 R1390: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x51_is_in_node_8 >= -1
 R1391: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x51_is_in_node_8 >= -1
 R1392: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_8 <= 1.1
 R1393: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x52_is_in_node_8 >= -1
 R1394: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x52_is_in_node_8 >= -1
 R1395: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_8 <= 1.1
 R1396: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x53_is_in_node_8 >= -1
 R1397: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x53_is_in_node_8 >= -1
 R1398: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_8 <= 1.1
 R1399: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.6 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x54_is_in_node_8 >= -1
 R1400: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x54_is_in_node_8 >= -1
 R1401: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_8 <= 1.1
 R1402: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x55_is_in_node_8 >= -1
 R1403: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x55_is_in_node_8 >= -1
 R1404: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_8 <= 1.1
 R1405: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x56_is_in_node_8 >= -1
 R1406: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x56_is_in_node_8 >= -1
 R1407: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_8 <= 1.1
 R1408: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 2.4 node_4_considers_var_1 + 3.3 node_4_considers_var_2
   + node_4_considers_var_3 - x57_is_in_node_8 >= -1
 R1409: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x57_is_in_node_8 >= -1
 R1410: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_8 <= 1.1
 R1411: - split_value_node_4 + 6.6 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.6 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x58_is_in_node_8 >= -1
 R1412: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x58_is_in_node_8 >= -1
 R1413: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_8 <= 1.1
 R1414: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 3.9 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x59_is_in_node_8 >= -1
 R1415: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x59_is_in_node_8 >= -1
 R1416: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_8 <= 1.1
 R1417: - split_value_node_4 + 5 node_4_considers_var_0
   + 2 node_4_considers_var_1 + 3.5 node_4_considers_var_2
   + node_4_considers_var_3 - x60_is_in_node_8 >= -1
 R1418: - split_value_node_2 + 5 node_2_considers_var_0
   + 2 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x60_is_in_node_8 >= -1
 R1419: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_8 <= 1.1
 R1420: - split_value_node_4 + 5.9 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x61_is_in_node_8 >= -1
 R1421: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x61_is_in_node_8 >= -1
 R1422: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_8 <= 1.1
 R1423: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.2 node_4_considers_var_1 + 4 node_4_considers_var_2
   + node_4_considers_var_3 - x62_is_in_node_8 >= -1
 R1424: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4 node_2_considers_var_2
   + node_2_considers_var_3 - x62_is_in_node_8 >= -1
 R1425: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_8 <= 1.1
 R1426: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x63_is_in_node_8 >= -1
 R1427: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x63_is_in_node_8 >= -1
 R1428: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_8 <= 1.1
 R1429: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 3.6 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x64_is_in_node_8 >= -1
 R1430: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 3.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x64_is_in_node_8 >= -1
 R1431: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_8 <= 1.1
 R1432: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x65_is_in_node_8 >= -1
 R1433: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x65_is_in_node_8 >= -1
 R1434: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_8 <= 1.1
 R1435: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x66_is_in_node_8 >= -1
 R1436: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x66_is_in_node_8 >= -1
 R1437: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_8 <= 1.1
 R1438: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 4.1 node_4_considers_var_2
   + node_4_considers_var_3 - x67_is_in_node_8 >= -1
 R1439: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + node_2_considers_var_3 - x67_is_in_node_8 >= -1
 R1440: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_8 <= 1.1
 R1441: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 2.2 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x68_is_in_node_8 >= -1
 R1442: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x68_is_in_node_8 >= -1
 R1443: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_8 <= 1.1
 R1444: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 3.9 node_4_considers_var_2
   + 1.1 node_4_considers_var_3 - x69_is_in_node_8 >= -1
 R1445: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x69_is_in_node_8 >= -1
 R1446: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_8 <= 1.1
 R1447: - split_value_node_4 + 5.9 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x70_is_in_node_8 >= -1
 R1448: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x70_is_in_node_8 >= -1
 R1449: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_8 <= 1.1
 R1450: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x71_is_in_node_8 >= -1
 R1451: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x71_is_in_node_8 >= -1
 R1452: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_8 <= 1.1
 R1453: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x72_is_in_node_8 >= -1
 R1454: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x72_is_in_node_8 >= -1
 R1455: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_8 <= 1.1
 R1456: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x73_is_in_node_8 >= -1
 R1457: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x73_is_in_node_8 >= -1
 R1458: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_8 <= 1.1
 R1459: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.3 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x74_is_in_node_8 >= -1
 R1460: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x74_is_in_node_8 >= -1
 R1461: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_8 <= 1.1
 R1462: - split_value_node_4 + 6.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x75_is_in_node_8 >= -1
 R1463: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x75_is_in_node_8 >= -1
 R1464: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_8 <= 1.1
 R1465: - split_value_node_4 + 6.8 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x76_is_in_node_8 >= -1
 R1466: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x76_is_in_node_8 >= -1
 R1467: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_8 <= 1.1
 R1468: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 1.7 node_4_considers_var_3 - x77_is_in_node_8 >= -1
 R1469: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x77_is_in_node_8 >= -1
 R1470: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_8 <= 1.1
 R1471: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x78_is_in_node_8 >= -1
 R1472: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x78_is_in_node_8 >= -1
 R1473: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_8 <= 1.1
 R1474: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 3.5 node_4_considers_var_2
   + node_4_considers_var_3 - x79_is_in_node_8 >= -1
 R1475: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x79_is_in_node_8 >= -1
 R1476: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_8 <= 1.1
 R1477: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.4 node_4_considers_var_1 + 3.8 node_4_considers_var_2
   + 1.1 node_4_considers_var_3 - x80_is_in_node_8 >= -1
 R1478: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.8 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x80_is_in_node_8 >= -1
 R1479: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_8 <= 1.1
 R1480: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.4 node_4_considers_var_1 + 3.7 node_4_considers_var_2
   + node_4_considers_var_3 - x81_is_in_node_8 >= -1
 R1481: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.7 node_2_considers_var_2
   + node_2_considers_var_3 - x81_is_in_node_8 >= -1
 R1482: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_8 <= 1.1
 R1483: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 3.9 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x82_is_in_node_8 >= -1
 R1484: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x82_is_in_node_8 >= -1
 R1485: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_8 <= 1.1
 R1486: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x83_is_in_node_8 >= -1
 R1487: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x83_is_in_node_8 >= -1
 R1488: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_8 <= 1.1
 R1489: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x84_is_in_node_8 >= -1
 R1490: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x84_is_in_node_8 >= -1
 R1491: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_8 <= 1.1
 R1492: - split_value_node_4 + 6 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x85_is_in_node_8 >= -1
 R1493: - split_value_node_2 + 6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x85_is_in_node_8 >= -1
 R1494: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_8 <= 1.1
 R1495: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x86_is_in_node_8 >= -1
 R1496: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x86_is_in_node_8 >= -1
 R1497: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_8 <= 1.1
 R1498: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x87_is_in_node_8 >= -1
 R1499: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x87_is_in_node_8 >= -1
 R1500: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_8 <= 1.1
 R1501: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.1 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x88_is_in_node_8 >= -1
 R1502: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x88_is_in_node_8 >= -1
 R1503: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_8 <= 1.1
 R1504: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x89_is_in_node_8 >= -1
 R1505: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x89_is_in_node_8 >= -1
 R1506: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_8 <= 1.1
 R1507: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x90_is_in_node_8 >= -1
 R1508: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x90_is_in_node_8 >= -1
 R1509: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_8 <= 1.1
 R1510: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.6 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x91_is_in_node_8 >= -1
 R1511: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x91_is_in_node_8 >= -1
 R1512: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_8 <= 1.1
 R1513: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x92_is_in_node_8 >= -1
 R1514: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x92_is_in_node_8 >= -1
 R1515: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_8 <= 1.1
 R1516: - split_value_node_4 + 5 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 3.3 node_4_considers_var_2
   + node_4_considers_var_3 - x93_is_in_node_8 >= -1
 R1517: - split_value_node_2 + 5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x93_is_in_node_8 >= -1
 R1518: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_8 <= 1.1
 R1519: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x94_is_in_node_8 >= -1
 R1520: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x94_is_in_node_8 >= -1
 R1521: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_8 <= 1.1
 R1522: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x95_is_in_node_8 >= -1
 R1523: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x95_is_in_node_8 >= -1
 R1524: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_8 <= 1.1
 R1525: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x96_is_in_node_8 >= -1
 R1526: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x96_is_in_node_8 >= -1
 R1527: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_8 <= 1.1
 R1528: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.3 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x97_is_in_node_8 >= -1
 R1529: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x97_is_in_node_8 >= -1
 R1530: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_8 <= 1.1
 R1531: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 3 node_4_considers_var_2
   + 1.1 node_4_considers_var_3 - x98_is_in_node_8 >= -1
 R1532: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x98_is_in_node_8 >= -1
 R1533: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_8 <= 1.1
 R1534: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.1 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x99_is_in_node_8 >= -1
 R1535: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x99_is_in_node_8 >= -1
 R1536: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_8 <= 1.1
 R1537: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 6 node_4_considers_var_2
   + 2.5 node_4_considers_var_3 - x100_is_in_node_8 >= -1
 R1538: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x100_is_in_node_8 >= -1
 R1539: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_8 <= 1.1
 R1540: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x101_is_in_node_8 >= -1
 R1541: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x101_is_in_node_8 >= -1
 R1542: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_8 <= 1.1
 R1543: - split_value_node_4 + 7.1 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.9 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x102_is_in_node_8 >= -1
 R1544: - split_value_node_2 + 7.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x102_is_in_node_8 >= -1
 R1545: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_8 <= 1.1
 R1546: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x103_is_in_node_8 >= -1
 R1547: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x103_is_in_node_8 >= -1
 R1548: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_8 <= 1.1
 R1549: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.8 node_4_considers_var_2
   + 2.2 node_4_considers_var_3 - x104_is_in_node_8 >= -1
 R1550: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x104_is_in_node_8 >= -1
 R1551: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_8 <= 1.1
 R1552: - split_value_node_4 + 7.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 6.6 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x105_is_in_node_8 >= -1
 R1553: - split_value_node_2 + 7.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x105_is_in_node_8 >= -1
 R1554: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_8 <= 1.1
 R1555: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.7 node_4_considers_var_3 - x106_is_in_node_8 >= -1
 R1556: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x106_is_in_node_8 >= -1
 R1557: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_8 <= 1.1
 R1558: - split_value_node_4 + 7.3 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 6.3 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x107_is_in_node_8 >= -1
 R1559: - split_value_node_2 + 7.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 6.3 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x107_is_in_node_8 >= -1
 R1560: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_8 <= 1.1
 R1561: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 5.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x108_is_in_node_8 >= -1
 R1562: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x108_is_in_node_8 >= -1
 R1563: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_8 <= 1.1
 R1564: - split_value_node_4 + 7.2 node_4_considers_var_0
   + 3.6 node_4_considers_var_1 + 6.1 node_4_considers_var_2
   + 2.5 node_4_considers_var_3 - x109_is_in_node_8 >= -1
 R1565: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x109_is_in_node_8 >= -1
 R1566: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_8 <= 1.1
 R1567: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x110_is_in_node_8 >= -1
 R1568: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x110_is_in_node_8 >= -1
 R1569: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_8 <= 1.1
 R1570: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.3 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x111_is_in_node_8 >= -1
 R1571: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x111_is_in_node_8 >= -1
 R1572: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_8 <= 1.1
 R1573: - split_value_node_4 + 6.8 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.5 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x112_is_in_node_8 >= -1
 R1574: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x112_is_in_node_8 >= -1
 R1575: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_8 <= 1.1
 R1576: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x113_is_in_node_8 >= -1
 R1577: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x113_is_in_node_8 >= -1
 R1578: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_8 <= 1.1
 R1579: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 2.4 node_4_considers_var_3 - x114_is_in_node_8 >= -1
 R1580: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x114_is_in_node_8 >= -1
 R1581: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_8 <= 1.1
 R1582: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.3 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x115_is_in_node_8 >= -1
 R1583: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x115_is_in_node_8 >= -1
 R1584: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_8 <= 1.1
 R1585: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.5 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x116_is_in_node_8 >= -1
 R1586: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x116_is_in_node_8 >= -1
 R1587: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_8 <= 1.1
 R1588: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 6.7 node_4_considers_var_2
   + 2.2 node_4_considers_var_3 - x117_is_in_node_8 >= -1
 R1589: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x117_is_in_node_8 >= -1
 R1590: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_8 <= 1.1
 R1591: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 6.9 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x118_is_in_node_8 >= -1
 R1592: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 6.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x118_is_in_node_8 >= -1
 R1593: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_8 <= 1.1
 R1594: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.2 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x119_is_in_node_8 >= -1
 R1595: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x119_is_in_node_8 >= -1
 R1596: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_8 <= 1.1
 R1597: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.7 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x120_is_in_node_8 >= -1
 R1598: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x120_is_in_node_8 >= -1
 R1599: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_8 <= 1.1
 R1600: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x121_is_in_node_8 >= -1
 R1601: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x121_is_in_node_8 >= -1
 R1602: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_8 <= 1.1
 R1603: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 6.7 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x122_is_in_node_8 >= -1
 R1604: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x122_is_in_node_8 >= -1
 R1605: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_8 <= 1.1
 R1606: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x123_is_in_node_8 >= -1
 R1607: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x123_is_in_node_8 >= -1
 R1608: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_8 <= 1.1
 R1609: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 5.7 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x124_is_in_node_8 >= -1
 R1610: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x124_is_in_node_8 >= -1
 R1611: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_8 <= 1.1
 R1612: - split_value_node_4 + 7.2 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 6 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x125_is_in_node_8 >= -1
 R1613: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x125_is_in_node_8 >= -1
 R1614: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_8 <= 1.1
 R1615: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x126_is_in_node_8 >= -1
 R1616: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x126_is_in_node_8 >= -1
 R1617: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_8 <= 1.1
 R1618: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x127_is_in_node_8 >= -1
 R1619: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x127_is_in_node_8 >= -1
 R1620: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_8 <= 1.1
 R1621: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x128_is_in_node_8 >= -1
 R1622: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x128_is_in_node_8 >= -1
 R1623: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_8 <= 1.1
 R1624: - split_value_node_4 + 7.2 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.8 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x129_is_in_node_8 >= -1
 R1625: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x129_is_in_node_8 >= -1
 R1626: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_8 <= 1.1
 R1627: - split_value_node_4 + 7.4 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 6.1 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x130_is_in_node_8 >= -1
 R1628: - split_value_node_2 + 7.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x130_is_in_node_8 >= -1
 R1629: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_8 <= 1.1
 R1630: - split_value_node_4 + 7.9 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 6.4 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x131_is_in_node_8 >= -1
 R1631: - split_value_node_2 + 7.9 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.4 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x131_is_in_node_8 >= -1
 R1632: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_8 <= 1.1
 R1633: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.2 node_4_considers_var_3 - x132_is_in_node_8 >= -1
 R1634: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x132_is_in_node_8 >= -1
 R1635: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_8 <= 1.1
 R1636: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x133_is_in_node_8 >= -1
 R1637: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x133_is_in_node_8 >= -1
 R1638: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_8 <= 1.1
 R1639: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x134_is_in_node_8 >= -1
 R1640: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x134_is_in_node_8 >= -1
 R1641: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_8 <= 1.1
 R1642: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 6.1 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x135_is_in_node_8 >= -1
 R1643: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x135_is_in_node_8 >= -1
 R1644: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_8 <= 1.1
 R1645: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.4 node_4_considers_var_3 - x136_is_in_node_8 >= -1
 R1646: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x136_is_in_node_8 >= -1
 R1647: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_8 <= 1.1
 R1648: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.5 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x137_is_in_node_8 >= -1
 R1649: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x137_is_in_node_8 >= -1
 R1650: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_8 <= 1.1
 R1651: - split_value_node_4 + 6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x138_is_in_node_8 >= -1
 R1652: - split_value_node_2 + 6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x138_is_in_node_8 >= -1
 R1653: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_8 <= 1.1
 R1654: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.4 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x139_is_in_node_8 >= -1
 R1655: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x139_is_in_node_8 >= -1
 R1656: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_8 <= 1.1
 R1657: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.4 node_4_considers_var_3 - x140_is_in_node_8 >= -1
 R1658: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x140_is_in_node_8 >= -1
 R1659: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_8 <= 1.1
 R1660: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x141_is_in_node_8 >= -1
 R1661: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x141_is_in_node_8 >= -1
 R1662: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_8 <= 1.1
 R1663: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x142_is_in_node_8 >= -1
 R1664: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x142_is_in_node_8 >= -1
 R1665: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_8 <= 1.1
 R1666: - split_value_node_4 + 6.8 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.9 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x143_is_in_node_8 >= -1
 R1667: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x143_is_in_node_8 >= -1
 R1668: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_8 <= 1.1
 R1669: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 5.7 node_4_considers_var_2
   + 2.5 node_4_considers_var_3 - x144_is_in_node_8 >= -1
 R1670: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x144_is_in_node_8 >= -1
 R1671: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_8 <= 1.1
 R1672: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.2 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x145_is_in_node_8 >= -1
 R1673: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x145_is_in_node_8 >= -1
 R1674: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_8 <= 1.1
 R1675: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x146_is_in_node_8 >= -1
 R1676: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x146_is_in_node_8 >= -1
 R1677: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_8 <= 1.1
 R1678: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.2 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x147_is_in_node_8 >= -1
 R1679: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x147_is_in_node_8 >= -1
 R1680: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_8 <= 1.1
 R1681: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 5.4 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x148_is_in_node_8 >= -1
 R1682: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x148_is_in_node_8 >= -1
 R1683: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_8 <= 1.1
 R1684: - split_value_node_4 + 5.9 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x149_is_in_node_8 >= -1
 R1685: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x149_is_in_node_8 >= -1
 R1686: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_9 <= 1.1
 R1687: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x0_is_in_node_9 >= -1
 R1688: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x0_is_in_node_9 >= -1
 R1689: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_9 <= 1.1
 R1690: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x1_is_in_node_9 >= -1
 R1691: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x1_is_in_node_9 >= -1
 R1692: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_9 <= 1.1
 R1693: - split_value_node_4 + 4.7 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x2_is_in_node_9 >= -1
 R1694: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x2_is_in_node_9 >= -1
 R1695: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_9 <= 1.1
 R1696: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x3_is_in_node_9 >= -1
 R1697: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x3_is_in_node_9 >= -1
 R1698: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_9 <= 1.1
 R1699: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.6 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x4_is_in_node_9 >= -1
 R1700: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x4_is_in_node_9 >= -1
 R1701: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_9 <= 1.1
 R1702: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.9 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x5_is_in_node_9 >= -1
 R1703: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x5_is_in_node_9 >= -1
 R1704: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_9 <= 1.1
 R1705: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x6_is_in_node_9 >= -1
 R1706: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x6_is_in_node_9 >= -1
 R1707: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_9 <= 1.1
 R1708: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x7_is_in_node_9 >= -1
 R1709: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x7_is_in_node_9 >= -1
 R1710: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_9 <= 1.1
 R1711: - split_value_node_4 + 4.4 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x8_is_in_node_9 >= -1
 R1712: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x8_is_in_node_9 >= -1
 R1713: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_9 <= 1.1
 R1714: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x9_is_in_node_9 >= -1
 R1715: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x9_is_in_node_9 >= -1
 R1716: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_9 <= 1.1
 R1717: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.7 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x10_is_in_node_9 >= -1
 R1718: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x10_is_in_node_9 >= -1
 R1719: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_9 <= 1.1
 R1720: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x11_is_in_node_9 >= -1
 R1721: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x11_is_in_node_9 >= -1
 R1722: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_9 <= 1.1
 R1723: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x12_is_in_node_9 >= -1
 R1724: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x12_is_in_node_9 >= -1
 R1725: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_9 <= 1.1
 R1726: - split_value_node_4 + 4.3 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.1 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x13_is_in_node_9 >= -1
 R1727: - split_value_node_2 + 4.3 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.1 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x13_is_in_node_9 >= -1
 R1728: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_9 <= 1.1
 R1729: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 4 node_4_considers_var_1 + 1.2 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x14_is_in_node_9 >= -1
 R1730: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 4 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x14_is_in_node_9 >= -1
 R1731: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_9 <= 1.1
 R1732: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 4.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x15_is_in_node_9 >= -1
 R1733: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 4.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x15_is_in_node_9 >= -1
 R1734: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_9 <= 1.1
 R1735: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.9 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x16_is_in_node_9 >= -1
 R1736: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x16_is_in_node_9 >= -1
 R1737: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_9 <= 1.1
 R1738: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x17_is_in_node_9 >= -1
 R1739: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x17_is_in_node_9 >= -1
 R1740: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_9 <= 1.1
 R1741: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x18_is_in_node_9 >= -1
 R1742: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x18_is_in_node_9 >= -1
 R1743: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_9 <= 1.1
 R1744: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x19_is_in_node_9 >= -1
 R1745: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x19_is_in_node_9 >= -1
 R1746: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_9 <= 1.1
 R1747: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x20_is_in_node_9 >= -1
 R1748: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x20_is_in_node_9 >= -1
 R1749: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_9 <= 1.1
 R1750: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.7 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x21_is_in_node_9 >= -1
 R1751: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x21_is_in_node_9 >= -1
 R1752: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_9 <= 1.1
 R1753: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.6 node_4_considers_var_1 + node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x22_is_in_node_9 >= -1
 R1754: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x22_is_in_node_9 >= -1
 R1755: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_9 <= 1.1
 R1756: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 1.7 node_4_considers_var_2
   + 0.5 node_4_considers_var_3 - x23_is_in_node_9 >= -1
 R1757: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.5 node_2_considers_var_3 - x23_is_in_node_9 >= -1
 R1758: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_9 <= 1.1
 R1759: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.9 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x24_is_in_node_9 >= -1
 R1760: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x24_is_in_node_9 >= -1
 R1761: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_9 <= 1.1
 R1762: - split_value_node_4 + 5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x25_is_in_node_9 >= -1
 R1763: - split_value_node_2 + 5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x25_is_in_node_9 >= -1
 R1764: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_9 <= 1.1
 R1765: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x26_is_in_node_9 >= -1
 R1766: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x26_is_in_node_9 >= -1
 R1767: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_9 <= 1.1
 R1768: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x27_is_in_node_9 >= -1
 R1769: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x27_is_in_node_9 >= -1
 R1770: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_9 <= 1.1
 R1771: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x28_is_in_node_9 >= -1
 R1772: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x28_is_in_node_9 >= -1
 R1773: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_9 <= 1.1
 R1774: - split_value_node_4 + 4.7 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x29_is_in_node_9 >= -1
 R1775: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x29_is_in_node_9 >= -1
 R1776: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_9 <= 1.1
 R1777: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x30_is_in_node_9 >= -1
 R1778: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x30_is_in_node_9 >= -1
 R1779: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_9 <= 1.1
 R1780: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x31_is_in_node_9 >= -1
 R1781: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x31_is_in_node_9 >= -1
 R1782: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_9 <= 1.1
 R1783: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 4.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x32_is_in_node_9 >= -1
 R1784: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 4.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x32_is_in_node_9 >= -1
 R1785: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_9 <= 1.1
 R1786: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 4.2 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x33_is_in_node_9 >= -1
 R1787: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 4.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x33_is_in_node_9 >= -1
 R1788: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_9 <= 1.1
 R1789: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x34_is_in_node_9 >= -1
 R1790: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x34_is_in_node_9 >= -1
 R1791: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_9 <= 1.1
 R1792: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.2 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x35_is_in_node_9 >= -1
 R1793: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x35_is_in_node_9 >= -1
 R1794: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_9 <= 1.1
 R1795: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x36_is_in_node_9 >= -1
 R1796: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x36_is_in_node_9 >= -1
 R1797: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_9 <= 1.1
 R1798: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.1 node_4_considers_var_3 - x37_is_in_node_9 >= -1
 R1799: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x37_is_in_node_9 >= -1
 R1800: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_9 <= 1.1
 R1801: - split_value_node_4 + 4.4 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x38_is_in_node_9 >= -1
 R1802: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x38_is_in_node_9 >= -1
 R1803: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_9 <= 1.1
 R1804: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x39_is_in_node_9 >= -1
 R1805: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x39_is_in_node_9 >= -1
 R1806: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_9 <= 1.1
 R1807: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x40_is_in_node_9 >= -1
 R1808: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x40_is_in_node_9 >= -1
 R1809: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_9 <= 1.1
 R1810: - split_value_node_4 + 4.5 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x41_is_in_node_9 >= -1
 R1811: - split_value_node_2 + 4.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x41_is_in_node_9 >= -1
 R1812: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_9 <= 1.1
 R1813: - split_value_node_4 + 4.4 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.3 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x42_is_in_node_9 >= -1
 R1814: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x42_is_in_node_9 >= -1
 R1815: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_9 <= 1.1
 R1816: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.5 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.6 node_4_considers_var_3 - x43_is_in_node_9 >= -1
 R1817: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.6 node_2_considers_var_3 - x43_is_in_node_9 >= -1
 R1818: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_9 <= 1.1
 R1819: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.9 node_4_considers_var_2
   + 0.4 node_4_considers_var_3 - x44_is_in_node_9 >= -1
 R1820: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x44_is_in_node_9 >= -1
 R1821: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_9 <= 1.1
 R1822: - split_value_node_4 + 4.8 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.3 node_4_considers_var_3 - x45_is_in_node_9 >= -1
 R1823: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x45_is_in_node_9 >= -1
 R1824: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_9 <= 1.1
 R1825: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 1.6 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x46_is_in_node_9 >= -1
 R1826: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x46_is_in_node_9 >= -1
 R1827: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_9 <= 1.1
 R1828: - split_value_node_4 + 4.6 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x47_is_in_node_9 >= -1
 R1829: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x47_is_in_node_9 >= -1
 R1830: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_9 <= 1.1
 R1831: - split_value_node_4 + 5.3 node_4_considers_var_0
   + 3.7 node_4_considers_var_1 + 1.5 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x48_is_in_node_9 >= -1
 R1832: - split_value_node_2 + 5.3 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x48_is_in_node_9 >= -1
 R1833: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_9 <= 1.1
 R1834: - split_value_node_4 + 5 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 1.4 node_4_considers_var_2
   + 0.2 node_4_considers_var_3 - x49_is_in_node_9 >= -1
 R1835: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x49_is_in_node_9 >= -1
 R1836: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_9 <= 1.1
 R1837: - split_value_node_4 + 7 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x50_is_in_node_9 >= -1
 R1838: - split_value_node_2 + 7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x50_is_in_node_9 >= -1
 R1839: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_9 <= 1.1
 R1840: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x51_is_in_node_9 >= -1
 R1841: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x51_is_in_node_9 >= -1
 R1842: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_9 <= 1.1
 R1843: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x52_is_in_node_9 >= -1
 R1844: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x52_is_in_node_9 >= -1
 R1845: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_9 <= 1.1
 R1846: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x53_is_in_node_9 >= -1
 R1847: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x53_is_in_node_9 >= -1
 R1848: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_9 <= 1.1
 R1849: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.6 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x54_is_in_node_9 >= -1
 R1850: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x54_is_in_node_9 >= -1
 R1851: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_9 <= 1.1
 R1852: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x55_is_in_node_9 >= -1
 R1853: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x55_is_in_node_9 >= -1
 R1854: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_9 <= 1.1
 R1855: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x56_is_in_node_9 >= -1
 R1856: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x56_is_in_node_9 >= -1
 R1857: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_9 <= 1.1
 R1858: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 2.4 node_4_considers_var_1 + 3.3 node_4_considers_var_2
   + node_4_considers_var_3 - x57_is_in_node_9 >= -1
 R1859: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x57_is_in_node_9 >= -1
 R1860: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_9 <= 1.1
 R1861: - split_value_node_4 + 6.6 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.6 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x58_is_in_node_9 >= -1
 R1862: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x58_is_in_node_9 >= -1
 R1863: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_9 <= 1.1
 R1864: - split_value_node_4 + 5.2 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 3.9 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x59_is_in_node_9 >= -1
 R1865: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x59_is_in_node_9 >= -1
 R1866: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_9 <= 1.1
 R1867: - split_value_node_4 + 5 node_4_considers_var_0
   + 2 node_4_considers_var_1 + 3.5 node_4_considers_var_2
   + node_4_considers_var_3 - x60_is_in_node_9 >= -1
 R1868: - split_value_node_2 + 5 node_2_considers_var_0
   + 2 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x60_is_in_node_9 >= -1
 R1869: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_9 <= 1.1
 R1870: - split_value_node_4 + 5.9 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x61_is_in_node_9 >= -1
 R1871: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x61_is_in_node_9 >= -1
 R1872: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_9 <= 1.1
 R1873: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.2 node_4_considers_var_1 + 4 node_4_considers_var_2
   + node_4_considers_var_3 - x62_is_in_node_9 >= -1
 R1874: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4 node_2_considers_var_2
   + node_2_considers_var_3 - x62_is_in_node_9 >= -1
 R1875: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_9 <= 1.1
 R1876: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x63_is_in_node_9 >= -1
 R1877: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x63_is_in_node_9 >= -1
 R1878: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_9 <= 1.1
 R1879: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 3.6 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x64_is_in_node_9 >= -1
 R1880: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 3.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x64_is_in_node_9 >= -1
 R1881: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_9 <= 1.1
 R1882: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x65_is_in_node_9 >= -1
 R1883: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x65_is_in_node_9 >= -1
 R1884: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_9 <= 1.1
 R1885: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x66_is_in_node_9 >= -1
 R1886: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x66_is_in_node_9 >= -1
 R1887: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_9 <= 1.1
 R1888: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 4.1 node_4_considers_var_2
   + node_4_considers_var_3 - x67_is_in_node_9 >= -1
 R1889: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + node_2_considers_var_3 - x67_is_in_node_9 >= -1
 R1890: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_9 <= 1.1
 R1891: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 2.2 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x68_is_in_node_9 >= -1
 R1892: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x68_is_in_node_9 >= -1
 R1893: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_9 <= 1.1
 R1894: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 3.9 node_4_considers_var_2
   + 1.1 node_4_considers_var_3 - x69_is_in_node_9 >= -1
 R1895: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x69_is_in_node_9 >= -1
 R1896: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_9 <= 1.1
 R1897: - split_value_node_4 + 5.9 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x70_is_in_node_9 >= -1
 R1898: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x70_is_in_node_9 >= -1
 R1899: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_9 <= 1.1
 R1900: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x71_is_in_node_9 >= -1
 R1901: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x71_is_in_node_9 >= -1
 R1902: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_9 <= 1.1
 R1903: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x72_is_in_node_9 >= -1
 R1904: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x72_is_in_node_9 >= -1
 R1905: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_9 <= 1.1
 R1906: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x73_is_in_node_9 >= -1
 R1907: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x73_is_in_node_9 >= -1
 R1908: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_9 <= 1.1
 R1909: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.3 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x74_is_in_node_9 >= -1
 R1910: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x74_is_in_node_9 >= -1
 R1911: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_9 <= 1.1
 R1912: - split_value_node_4 + 6.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x75_is_in_node_9 >= -1
 R1913: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x75_is_in_node_9 >= -1
 R1914: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_9 <= 1.1
 R1915: - split_value_node_4 + 6.8 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x76_is_in_node_9 >= -1
 R1916: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x76_is_in_node_9 >= -1
 R1917: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_9 <= 1.1
 R1918: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 1.7 node_4_considers_var_3 - x77_is_in_node_9 >= -1
 R1919: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x77_is_in_node_9 >= -1
 R1920: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_9 <= 1.1
 R1921: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x78_is_in_node_9 >= -1
 R1922: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x78_is_in_node_9 >= -1
 R1923: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_9 <= 1.1
 R1924: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 3.5 node_4_considers_var_2
   + node_4_considers_var_3 - x79_is_in_node_9 >= -1
 R1925: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x79_is_in_node_9 >= -1
 R1926: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_9 <= 1.1
 R1927: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.4 node_4_considers_var_1 + 3.8 node_4_considers_var_2
   + 1.1 node_4_considers_var_3 - x80_is_in_node_9 >= -1
 R1928: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.8 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x80_is_in_node_9 >= -1
 R1929: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_9 <= 1.1
 R1930: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.4 node_4_considers_var_1 + 3.7 node_4_considers_var_2
   + node_4_considers_var_3 - x81_is_in_node_9 >= -1
 R1931: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.7 node_2_considers_var_2
   + node_2_considers_var_3 - x81_is_in_node_9 >= -1
 R1932: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_9 <= 1.1
 R1933: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 3.9 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x82_is_in_node_9 >= -1
 R1934: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x82_is_in_node_9 >= -1
 R1935: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_9 <= 1.1
 R1936: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x83_is_in_node_9 >= -1
 R1937: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x83_is_in_node_9 >= -1
 R1938: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_9 <= 1.1
 R1939: - split_value_node_4 + 5.4 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x84_is_in_node_9 >= -1
 R1940: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x84_is_in_node_9 >= -1
 R1941: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_9 <= 1.1
 R1942: - split_value_node_4 + 6 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x85_is_in_node_9 >= -1
 R1943: - split_value_node_2 + 6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x85_is_in_node_9 >= -1
 R1944: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_9 <= 1.1
 R1945: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 4.7 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x86_is_in_node_9 >= -1
 R1946: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x86_is_in_node_9 >= -1
 R1947: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_9 <= 1.1
 R1948: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x87_is_in_node_9 >= -1
 R1949: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x87_is_in_node_9 >= -1
 R1950: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_9 <= 1.1
 R1951: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.1 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x88_is_in_node_9 >= -1
 R1952: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x88_is_in_node_9 >= -1
 R1953: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_9 <= 1.1
 R1954: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x89_is_in_node_9 >= -1
 R1955: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x89_is_in_node_9 >= -1
 R1956: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_9 <= 1.1
 R1957: - split_value_node_4 + 5.5 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 4.4 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x90_is_in_node_9 >= -1
 R1958: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x90_is_in_node_9 >= -1
 R1959: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_9 <= 1.1
 R1960: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.6 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x91_is_in_node_9 >= -1
 R1961: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x91_is_in_node_9 >= -1
 R1962: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_9 <= 1.1
 R1963: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 4 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x92_is_in_node_9 >= -1
 R1964: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x92_is_in_node_9 >= -1
 R1965: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_9 <= 1.1
 R1966: - split_value_node_4 + 5 node_4_considers_var_0
   + 2.3 node_4_considers_var_1 + 3.3 node_4_considers_var_2
   + node_4_considers_var_3 - x93_is_in_node_9 >= -1
 R1967: - split_value_node_2 + 5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x93_is_in_node_9 >= -1
 R1968: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_9 <= 1.1
 R1969: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x94_is_in_node_9 >= -1
 R1970: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x94_is_in_node_9 >= -1
 R1971: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_9 <= 1.1
 R1972: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.2 node_4_considers_var_3 - x95_is_in_node_9 >= -1
 R1973: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x95_is_in_node_9 >= -1
 R1974: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_9 <= 1.1
 R1975: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.2 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x96_is_in_node_9 >= -1
 R1976: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x96_is_in_node_9 >= -1
 R1977: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_9 <= 1.1
 R1978: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 4.3 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x97_is_in_node_9 >= -1
 R1979: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x97_is_in_node_9 >= -1
 R1980: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_9 <= 1.1
 R1981: - split_value_node_4 + 5.1 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 3 node_4_considers_var_2
   + 1.1 node_4_considers_var_3 - x98_is_in_node_9 >= -1
 R1982: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x98_is_in_node_9 >= -1
 R1983: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_9 <= 1.1
 R1984: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.1 node_4_considers_var_2
   + 1.3 node_4_considers_var_3 - x99_is_in_node_9 >= -1
 R1985: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x99_is_in_node_9 >= -1
 R1986: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_9 <= 1.1
 R1987: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 6 node_4_considers_var_2
   + 2.5 node_4_considers_var_3 - x100_is_in_node_9 >= -1
 R1988: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x100_is_in_node_9 >= -1
 R1989: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_9 <= 1.1
 R1990: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x101_is_in_node_9 >= -1
 R1991: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x101_is_in_node_9 >= -1
 R1992: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_9 <= 1.1
 R1993: - split_value_node_4 + 7.1 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.9 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x102_is_in_node_9 >= -1
 R1994: - split_value_node_2 + 7.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x102_is_in_node_9 >= -1
 R1995: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_9 <= 1.1
 R1996: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x103_is_in_node_9 >= -1
 R1997: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x103_is_in_node_9 >= -1
 R1998: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_9 <= 1.1
 R1999: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.8 node_4_considers_var_2
   + 2.2 node_4_considers_var_3 - x104_is_in_node_9 >= -1
 R2000: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x104_is_in_node_9 >= -1
 R2001: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_9 <= 1.1
 R2002: - split_value_node_4 + 7.6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 6.6 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x105_is_in_node_9 >= -1
 R2003: - split_value_node_2 + 7.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x105_is_in_node_9 >= -1
 R2004: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_9 <= 1.1
 R2005: - split_value_node_4 + 4.9 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 4.5 node_4_considers_var_2
   + 1.7 node_4_considers_var_3 - x106_is_in_node_9 >= -1
 R2006: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x106_is_in_node_9 >= -1
 R2007: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_9 <= 1.1
 R2008: - split_value_node_4 + 7.3 node_4_considers_var_0
   + 2.9 node_4_considers_var_1 + 6.3 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x107_is_in_node_9 >= -1
 R2009: - split_value_node_2 + 7.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 6.3 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x107_is_in_node_9 >= -1
 R2010: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_9 <= 1.1
 R2011: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 5.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x108_is_in_node_9 >= -1
 R2012: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x108_is_in_node_9 >= -1
 R2013: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_9 <= 1.1
 R2014: - split_value_node_4 + 7.2 node_4_considers_var_0
   + 3.6 node_4_considers_var_1 + 6.1 node_4_considers_var_2
   + 2.5 node_4_considers_var_3 - x109_is_in_node_9 >= -1
 R2015: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x109_is_in_node_9 >= -1
 R2016: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_9 <= 1.1
 R2017: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x110_is_in_node_9 >= -1
 R2018: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x110_is_in_node_9 >= -1
 R2019: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_9 <= 1.1
 R2020: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.3 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x111_is_in_node_9 >= -1
 R2021: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x111_is_in_node_9 >= -1
 R2022: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_9 <= 1.1
 R2023: - split_value_node_4 + 6.8 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.5 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x112_is_in_node_9 >= -1
 R2024: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x112_is_in_node_9 >= -1
 R2025: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_9 <= 1.1
 R2026: - split_value_node_4 + 5.7 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x113_is_in_node_9 >= -1
 R2027: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x113_is_in_node_9 >= -1
 R2028: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_9 <= 1.1
 R2029: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 2.4 node_4_considers_var_3 - x114_is_in_node_9 >= -1
 R2030: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x114_is_in_node_9 >= -1
 R2031: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_9 <= 1.1
 R2032: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.3 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x115_is_in_node_9 >= -1
 R2033: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x115_is_in_node_9 >= -1
 R2034: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_9 <= 1.1
 R2035: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.5 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x116_is_in_node_9 >= -1
 R2036: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x116_is_in_node_9 >= -1
 R2037: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_9 <= 1.1
 R2038: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 6.7 node_4_considers_var_2
   + 2.2 node_4_considers_var_3 - x117_is_in_node_9 >= -1
 R2039: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x117_is_in_node_9 >= -1
 R2040: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_9 <= 1.1
 R2041: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 6.9 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x118_is_in_node_9 >= -1
 R2042: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 6.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x118_is_in_node_9 >= -1
 R2043: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_9 <= 1.1
 R2044: - split_value_node_4 + 6 node_4_considers_var_0
   + 2.2 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x119_is_in_node_9 >= -1
 R2045: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x119_is_in_node_9 >= -1
 R2046: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_9 <= 1.1
 R2047: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.7 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x120_is_in_node_9 >= -1
 R2048: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x120_is_in_node_9 >= -1
 R2049: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_9 <= 1.1
 R2050: - split_value_node_4 + 5.6 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x121_is_in_node_9 >= -1
 R2051: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x121_is_in_node_9 >= -1
 R2052: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_9 <= 1.1
 R2053: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 6.7 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x122_is_in_node_9 >= -1
 R2054: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x122_is_in_node_9 >= -1
 R2055: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_9 <= 1.1
 R2056: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x123_is_in_node_9 >= -1
 R2057: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x123_is_in_node_9 >= -1
 R2058: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_9 <= 1.1
 R2059: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 5.7 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x124_is_in_node_9 >= -1
 R2060: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x124_is_in_node_9 >= -1
 R2061: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_9 <= 1.1
 R2062: - split_value_node_4 + 7.2 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 6 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x125_is_in_node_9 >= -1
 R2063: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x125_is_in_node_9 >= -1
 R2064: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_9 <= 1.1
 R2065: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x126_is_in_node_9 >= -1
 R2066: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x126_is_in_node_9 >= -1
 R2067: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_9 <= 1.1
 R2068: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.9 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x127_is_in_node_9 >= -1
 R2069: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x127_is_in_node_9 >= -1
 R2070: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_9 <= 1.1
 R2071: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x128_is_in_node_9 >= -1
 R2072: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x128_is_in_node_9 >= -1
 R2073: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_9 <= 1.1
 R2074: - split_value_node_4 + 7.2 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.8 node_4_considers_var_2
   + 1.6 node_4_considers_var_3 - x129_is_in_node_9 >= -1
 R2075: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x129_is_in_node_9 >= -1
 R2076: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_9 <= 1.1
 R2077: - split_value_node_4 + 7.4 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 6.1 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x130_is_in_node_9 >= -1
 R2078: - split_value_node_2 + 7.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x130_is_in_node_9 >= -1
 R2079: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_9 <= 1.1
 R2080: - split_value_node_4 + 7.9 node_4_considers_var_0
   + 3.8 node_4_considers_var_1 + 6.4 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x131_is_in_node_9 >= -1
 R2081: - split_value_node_2 + 7.9 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.4 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x131_is_in_node_9 >= -1
 R2082: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_9 <= 1.1
 R2083: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.2 node_4_considers_var_3 - x132_is_in_node_9 >= -1
 R2084: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x132_is_in_node_9 >= -1
 R2085: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_9 <= 1.1
 R2086: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.8 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.5 node_4_considers_var_3 - x133_is_in_node_9 >= -1
 R2087: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x133_is_in_node_9 >= -1
 R2088: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_9 <= 1.1
 R2089: - split_value_node_4 + 6.1 node_4_considers_var_0
   + 2.6 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 1.4 node_4_considers_var_3 - x134_is_in_node_9 >= -1
 R2090: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x134_is_in_node_9 >= -1
 R2091: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_9 <= 1.1
 R2092: - split_value_node_4 + 7.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 6.1 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x135_is_in_node_9 >= -1
 R2093: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x135_is_in_node_9 >= -1
 R2094: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_9 <= 1.1
 R2095: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.4 node_4_considers_var_3 - x136_is_in_node_9 >= -1
 R2096: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x136_is_in_node_9 >= -1
 R2097: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_9 <= 1.1
 R2098: - split_value_node_4 + 6.4 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.5 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x137_is_in_node_9 >= -1
 R2099: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x137_is_in_node_9 >= -1
 R2100: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_9 <= 1.1
 R2101: - split_value_node_4 + 6 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 4.8 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x138_is_in_node_9 >= -1
 R2102: - split_value_node_2 + 6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x138_is_in_node_9 >= -1
 R2103: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_9 <= 1.1
 R2104: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.4 node_4_considers_var_2
   + 2.1 node_4_considers_var_3 - x139_is_in_node_9 >= -1
 R2105: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x139_is_in_node_9 >= -1
 R2106: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_9 <= 1.1
 R2107: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.6 node_4_considers_var_2
   + 2.4 node_4_considers_var_3 - x140_is_in_node_9 >= -1
 R2108: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x140_is_in_node_9 >= -1
 R2109: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_9 <= 1.1
 R2110: - split_value_node_4 + 6.9 node_4_considers_var_0
   + 3.1 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x141_is_in_node_9 >= -1
 R2111: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x141_is_in_node_9 >= -1
 R2112: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_9 <= 1.1
 R2113: - split_value_node_4 + 5.8 node_4_considers_var_0
   + 2.7 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x142_is_in_node_9 >= -1
 R2114: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x142_is_in_node_9 >= -1
 R2115: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_9 <= 1.1
 R2116: - split_value_node_4 + 6.8 node_4_considers_var_0
   + 3.2 node_4_considers_var_1 + 5.9 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x143_is_in_node_9 >= -1
 R2117: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x143_is_in_node_9 >= -1
 R2118: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_9 <= 1.1
 R2119: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3.3 node_4_considers_var_1 + 5.7 node_4_considers_var_2
   + 2.5 node_4_considers_var_3 - x144_is_in_node_9 >= -1
 R2120: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x144_is_in_node_9 >= -1
 R2121: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_9 <= 1.1
 R2122: - split_value_node_4 + 6.7 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.2 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x145_is_in_node_9 >= -1
 R2123: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x145_is_in_node_9 >= -1
 R2124: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_9 <= 1.1
 R2125: - split_value_node_4 + 6.3 node_4_considers_var_0
   + 2.5 node_4_considers_var_1 + 5 node_4_considers_var_2
   + 1.9 node_4_considers_var_3 - x146_is_in_node_9 >= -1
 R2126: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x146_is_in_node_9 >= -1
 R2127: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_9 <= 1.1
 R2128: - split_value_node_4 + 6.5 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.2 node_4_considers_var_2
   + 2 node_4_considers_var_3 - x147_is_in_node_9 >= -1
 R2129: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x147_is_in_node_9 >= -1
 R2130: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_9 <= 1.1
 R2131: - split_value_node_4 + 6.2 node_4_considers_var_0
   + 3.4 node_4_considers_var_1 + 5.4 node_4_considers_var_2
   + 2.3 node_4_considers_var_3 - x148_is_in_node_9 >= -1
 R2132: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x148_is_in_node_9 >= -1
 R2133: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_9 <= 1.1
 R2134: - split_value_node_4 + 5.9 node_4_considers_var_0
   + 3 node_4_considers_var_1 + 5.1 node_4_considers_var_2
   + 1.8 node_4_considers_var_3 - x149_is_in_node_9 >= -1
 R2135: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x149_is_in_node_9 >= -1
 R2136: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x0_is_in_node_10 <= 1.1
 R2137: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_10 <= 1.1
 R2138: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x0_is_in_node_10 >= -1
 R2139: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x1_is_in_node_10 <= 1.1
 R2140: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_10 <= 1.1
 R2141: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x1_is_in_node_10 >= -1
 R2142: - split_value_node_5 + 4.8 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x2_is_in_node_10 <= 1.1
 R2143: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_10 <= 1.1
 R2144: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x2_is_in_node_10 >= -1
 R2145: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x3_is_in_node_10 <= 1.1
 R2146: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_10 <= 1.1
 R2147: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x3_is_in_node_10 >= -1
 R2148: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.7 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x4_is_in_node_10 <= 1.1
 R2149: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_10 <= 1.1
 R2150: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x4_is_in_node_10 >= -1
 R2151: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 4 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x5_is_in_node_10 <= 1.1
 R2152: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_10 <= 1.1
 R2153: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x5_is_in_node_10 >= -1
 R2154: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x6_is_in_node_10 <= 1.1
 R2155: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_10 <= 1.1
 R2156: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x6_is_in_node_10 >= -1
 R2157: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x7_is_in_node_10 <= 1.1
 R2158: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_10 <= 1.1
 R2159: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x7_is_in_node_10 >= -1
 R2160: - split_value_node_5 + 4.5 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x8_is_in_node_10 <= 1.1
 R2161: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_10 <= 1.1
 R2162: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x8_is_in_node_10 >= -1
 R2163: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x9_is_in_node_10 <= 1.1
 R2164: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_10 <= 1.1
 R2165: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x9_is_in_node_10 >= -1
 R2166: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.8 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x10_is_in_node_10 <= 1.1
 R2167: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_10 <= 1.1
 R2168: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x10_is_in_node_10 >= -1
 R2169: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x11_is_in_node_10 <= 1.1
 R2170: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_10 <= 1.1
 R2171: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x11_is_in_node_10 >= -1
 R2172: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x12_is_in_node_10 <= 1.1
 R2173: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_10 <= 1.1
 R2174: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x12_is_in_node_10 >= -1
 R2175: - split_value_node_5 + 4.4 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.2 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x13_is_in_node_10 <= 1.1
 R2176: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_10 <= 1.1
 R2177: - split_value_node_2 + 4.3 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.1 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x13_is_in_node_10 >= -1
 R2178: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 4.1 node_5_considers_var_1 + 1.3 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x14_is_in_node_10 <= 1.1
 R2179: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_10 <= 1.1
 R2180: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 4 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x14_is_in_node_10 >= -1
 R2181: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 4.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x15_is_in_node_10 <= 1.1
 R2182: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_10 <= 1.1
 R2183: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 4.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x15_is_in_node_10 >= -1
 R2184: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 4 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x16_is_in_node_10 <= 1.1
 R2185: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_10 <= 1.1
 R2186: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x16_is_in_node_10 >= -1
 R2187: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x17_is_in_node_10 <= 1.1
 R2188: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_10 <= 1.1
 R2189: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x17_is_in_node_10 >= -1
 R2190: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x18_is_in_node_10 <= 1.1
 R2191: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_10 <= 1.1
 R2192: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x18_is_in_node_10 >= -1
 R2193: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x19_is_in_node_10 <= 1.1
 R2194: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_10 <= 1.1
 R2195: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x19_is_in_node_10 >= -1
 R2196: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x20_is_in_node_10 <= 1.1
 R2197: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_10 <= 1.1
 R2198: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x20_is_in_node_10 >= -1
 R2199: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.8 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x21_is_in_node_10 <= 1.1
 R2200: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_10 <= 1.1
 R2201: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x21_is_in_node_10 >= -1
 R2202: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.7 node_5_considers_var_1 + 1.1 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x22_is_in_node_10 <= 1.1
 R2203: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_10 <= 1.1
 R2204: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x22_is_in_node_10 >= -1
 R2205: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.6 node_5_considers_var_3 + 1.1 x23_is_in_node_10 <= 1.1
 R2206: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_10 <= 1.1
 R2207: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.5 node_2_considers_var_3 - x23_is_in_node_10 >= -1
 R2208: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 2 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x24_is_in_node_10 <= 1.1
 R2209: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_10 <= 1.1
 R2210: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x24_is_in_node_10 >= -1
 R2211: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x25_is_in_node_10 <= 1.1
 R2212: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_10 <= 1.1
 R2213: - split_value_node_2 + 5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x25_is_in_node_10 >= -1
 R2214: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x26_is_in_node_10 <= 1.1
 R2215: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_10 <= 1.1
 R2216: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x26_is_in_node_10 >= -1
 R2217: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x27_is_in_node_10 <= 1.1
 R2218: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_10 <= 1.1
 R2219: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x27_is_in_node_10 >= -1
 R2220: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x28_is_in_node_10 <= 1.1
 R2221: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_10 <= 1.1
 R2222: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x28_is_in_node_10 >= -1
 R2223: - split_value_node_5 + 4.8 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x29_is_in_node_10 <= 1.1
 R2224: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_10 <= 1.1
 R2225: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x29_is_in_node_10 >= -1
 R2226: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x30_is_in_node_10 <= 1.1
 R2227: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_10 <= 1.1
 R2228: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x30_is_in_node_10 >= -1
 R2229: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x31_is_in_node_10 <= 1.1
 R2230: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_10 <= 1.1
 R2231: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x31_is_in_node_10 >= -1
 R2232: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 4.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x32_is_in_node_10 <= 1.1
 R2233: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_10 <= 1.1
 R2234: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 4.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x32_is_in_node_10 >= -1
 R2235: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 4.3 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x33_is_in_node_10 <= 1.1
 R2236: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_10 <= 1.1
 R2237: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 4.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x33_is_in_node_10 >= -1
 R2238: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x34_is_in_node_10 <= 1.1
 R2239: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_10 <= 1.1
 R2240: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x34_is_in_node_10 >= -1
 R2241: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.3 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x35_is_in_node_10 <= 1.1
 R2242: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_10 <= 1.1
 R2243: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x35_is_in_node_10 >= -1
 R2244: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x36_is_in_node_10 <= 1.1
 R2245: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_10 <= 1.1
 R2246: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x36_is_in_node_10 >= -1
 R2247: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x37_is_in_node_10 <= 1.1
 R2248: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_10 <= 1.1
 R2249: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x37_is_in_node_10 >= -1
 R2250: - split_value_node_5 + 4.5 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x38_is_in_node_10 <= 1.1
 R2251: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_10 <= 1.1
 R2252: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x38_is_in_node_10 >= -1
 R2253: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x39_is_in_node_10 <= 1.1
 R2254: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_10 <= 1.1
 R2255: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x39_is_in_node_10 >= -1
 R2256: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x40_is_in_node_10 <= 1.1
 R2257: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_10 <= 1.1
 R2258: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x40_is_in_node_10 >= -1
 R2259: - split_value_node_5 + 4.6 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x41_is_in_node_10 <= 1.1
 R2260: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_10 <= 1.1
 R2261: - split_value_node_2 + 4.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x41_is_in_node_10 >= -1
 R2262: - split_value_node_5 + 4.5 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x42_is_in_node_10 <= 1.1
 R2263: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_10 <= 1.1
 R2264: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x42_is_in_node_10 >= -1
 R2265: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.7 node_5_considers_var_3 + 1.1 x43_is_in_node_10 <= 1.1
 R2266: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_10 <= 1.1
 R2267: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.6 node_2_considers_var_3 - x43_is_in_node_10 >= -1
 R2268: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 2 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x44_is_in_node_10 <= 1.1
 R2269: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_10 <= 1.1
 R2270: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x44_is_in_node_10 >= -1
 R2271: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x45_is_in_node_10 <= 1.1
 R2272: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_10 <= 1.1
 R2273: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x45_is_in_node_10 >= -1
 R2274: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x46_is_in_node_10 <= 1.1
 R2275: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_10 <= 1.1
 R2276: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x46_is_in_node_10 >= -1
 R2277: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x47_is_in_node_10 <= 1.1
 R2278: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_10 <= 1.1
 R2279: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x47_is_in_node_10 >= -1
 R2280: - split_value_node_5 + 5.4 node_5_considers_var_0
   + 3.8 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x48_is_in_node_10 <= 1.1
 R2281: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_10 <= 1.1
 R2282: - split_value_node_2 + 5.3 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x48_is_in_node_10 >= -1
 R2283: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x49_is_in_node_10 <= 1.1
 R2284: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_10 <= 1.1
 R2285: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x49_is_in_node_10 >= -1
 R2286: - split_value_node_5 + 7.1 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x50_is_in_node_10 <= 1.1
 R2287: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_10 <= 1.1
 R2288: - split_value_node_2 + 7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x50_is_in_node_10 >= -1
 R2289: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x51_is_in_node_10 <= 1.1
 R2290: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_10 <= 1.1
 R2291: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x51_is_in_node_10 >= -1
 R2292: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x52_is_in_node_10 <= 1.1
 R2293: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_10 <= 1.1
 R2294: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x52_is_in_node_10 >= -1
 R2295: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x53_is_in_node_10 <= 1.1
 R2296: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_10 <= 1.1
 R2297: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x53_is_in_node_10 >= -1
 R2298: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.7 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x54_is_in_node_10 <= 1.1
 R2299: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_10 <= 1.1
 R2300: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x54_is_in_node_10 >= -1
 R2301: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x55_is_in_node_10 <= 1.1
 R2302: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_10 <= 1.1
 R2303: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x55_is_in_node_10 >= -1
 R2304: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x56_is_in_node_10 <= 1.1
 R2305: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_10 <= 1.1
 R2306: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x56_is_in_node_10 >= -1
 R2307: - split_value_node_5 + 5 node_5_considers_var_0
   + 2.5 node_5_considers_var_1 + 3.4 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x57_is_in_node_10 <= 1.1
 R2308: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_10 <= 1.1
 R2309: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x57_is_in_node_10 >= -1
 R2310: - split_value_node_5 + 6.7 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.7 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x58_is_in_node_10 <= 1.1
 R2311: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_10 <= 1.1
 R2312: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x58_is_in_node_10 >= -1
 R2313: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x59_is_in_node_10 <= 1.1
 R2314: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_10 <= 1.1
 R2315: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x59_is_in_node_10 >= -1
 R2316: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 2.1 node_5_considers_var_1 + 3.6 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x60_is_in_node_10 <= 1.1
 R2317: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_10 <= 1.1
 R2318: - split_value_node_2 + 5 node_2_considers_var_0
   + 2 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x60_is_in_node_10 >= -1
 R2319: - split_value_node_5 + 6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x61_is_in_node_10 <= 1.1
 R2320: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_10 <= 1.1
 R2321: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x61_is_in_node_10 >= -1
 R2322: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 2.3 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x62_is_in_node_10 <= 1.1
 R2323: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_10 <= 1.1
 R2324: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4 node_2_considers_var_2
   + node_2_considers_var_3 - x62_is_in_node_10 >= -1
 R2325: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x63_is_in_node_10 <= 1.1
 R2326: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_10 <= 1.1
 R2327: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x63_is_in_node_10 >= -1
 R2328: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 3.7 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x64_is_in_node_10 <= 1.1
 R2329: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_10 <= 1.1
 R2330: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 3.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x64_is_in_node_10 >= -1
 R2331: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x65_is_in_node_10 <= 1.1
 R2332: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_10 <= 1.1
 R2333: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x65_is_in_node_10 >= -1
 R2334: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x66_is_in_node_10 <= 1.1
 R2335: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_10 <= 1.1
 R2336: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x66_is_in_node_10 >= -1
 R2337: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4.2 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x67_is_in_node_10 <= 1.1
 R2338: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_10 <= 1.1
 R2339: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + node_2_considers_var_3 - x67_is_in_node_10 >= -1
 R2340: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 2.3 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x68_is_in_node_10 <= 1.1
 R2341: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_10 <= 1.1
 R2342: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x68_is_in_node_10 >= -1
 R2343: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 4 node_5_considers_var_2
   + 1.2 node_5_considers_var_3 + 1.1 x69_is_in_node_10 <= 1.1
 R2344: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_10 <= 1.1
 R2345: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x69_is_in_node_10 >= -1
 R2346: - split_value_node_5 + 6 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x70_is_in_node_10 <= 1.1
 R2347: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_10 <= 1.1
 R2348: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x70_is_in_node_10 >= -1
 R2349: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x71_is_in_node_10 <= 1.1
 R2350: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_10 <= 1.1
 R2351: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x71_is_in_node_10 >= -1
 R2352: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x72_is_in_node_10 <= 1.1
 R2353: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_10 <= 1.1
 R2354: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x72_is_in_node_10 >= -1
 R2355: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x73_is_in_node_10 <= 1.1
 R2356: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_10 <= 1.1
 R2357: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x73_is_in_node_10 >= -1
 R2358: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.4 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x74_is_in_node_10 <= 1.1
 R2359: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_10 <= 1.1
 R2360: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x74_is_in_node_10 >= -1
 R2361: - split_value_node_5 + 6.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x75_is_in_node_10 <= 1.1
 R2362: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_10 <= 1.1
 R2363: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x75_is_in_node_10 >= -1
 R2364: - split_value_node_5 + 6.9 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x76_is_in_node_10 <= 1.1
 R2365: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_10 <= 1.1
 R2366: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x76_is_in_node_10 >= -1
 R2367: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 1.8 node_5_considers_var_3 + 1.1 x77_is_in_node_10 <= 1.1
 R2368: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_10 <= 1.1
 R2369: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x77_is_in_node_10 >= -1
 R2370: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x78_is_in_node_10 <= 1.1
 R2371: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_10 <= 1.1
 R2372: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x78_is_in_node_10 >= -1
 R2373: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 3.6 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x79_is_in_node_10 <= 1.1
 R2374: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_10 <= 1.1
 R2375: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x79_is_in_node_10 >= -1
 R2376: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.5 node_5_considers_var_1 + 3.9 node_5_considers_var_2
   + 1.2 node_5_considers_var_3 + 1.1 x80_is_in_node_10 <= 1.1
 R2377: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_10 <= 1.1
 R2378: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.8 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x80_is_in_node_10 >= -1
 R2379: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.5 node_5_considers_var_1 + 3.8 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x81_is_in_node_10 <= 1.1
 R2380: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_10 <= 1.1
 R2381: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.7 node_2_considers_var_2
   + node_2_considers_var_3 - x81_is_in_node_10 >= -1
 R2382: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x82_is_in_node_10 <= 1.1
 R2383: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_10 <= 1.1
 R2384: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x82_is_in_node_10 >= -1
 R2385: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x83_is_in_node_10 <= 1.1
 R2386: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_10 <= 1.1
 R2387: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x83_is_in_node_10 >= -1
 R2388: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x84_is_in_node_10 <= 1.1
 R2389: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_10 <= 1.1
 R2390: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x84_is_in_node_10 >= -1
 R2391: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x85_is_in_node_10 <= 1.1
 R2392: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_10 <= 1.1
 R2393: - split_value_node_2 + 6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x85_is_in_node_10 >= -1
 R2394: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x86_is_in_node_10 <= 1.1
 R2395: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_10 <= 1.1
 R2396: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x86_is_in_node_10 >= -1
 R2397: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x87_is_in_node_10 <= 1.1
 R2398: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_10 <= 1.1
 R2399: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x87_is_in_node_10 >= -1
 R2400: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.2 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x88_is_in_node_10 <= 1.1
 R2401: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_10 <= 1.1
 R2402: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x88_is_in_node_10 >= -1
 R2403: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x89_is_in_node_10 <= 1.1
 R2404: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_10 <= 1.1
 R2405: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x89_is_in_node_10 >= -1
 R2406: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x90_is_in_node_10 <= 1.1
 R2407: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_10 <= 1.1
 R2408: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x90_is_in_node_10 >= -1
 R2409: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.7 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x91_is_in_node_10 <= 1.1
 R2410: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_10 <= 1.1
 R2411: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x91_is_in_node_10 >= -1
 R2412: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x92_is_in_node_10 <= 1.1
 R2413: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_10 <= 1.1
 R2414: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x92_is_in_node_10 >= -1
 R2415: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 3.4 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x93_is_in_node_10 <= 1.1
 R2416: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_10 <= 1.1
 R2417: - split_value_node_2 + 5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x93_is_in_node_10 >= -1
 R2418: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x94_is_in_node_10 <= 1.1
 R2419: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_10 <= 1.1
 R2420: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x94_is_in_node_10 >= -1
 R2421: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x95_is_in_node_10 <= 1.1
 R2422: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_10 <= 1.1
 R2423: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x95_is_in_node_10 >= -1
 R2424: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x96_is_in_node_10 <= 1.1
 R2425: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_10 <= 1.1
 R2426: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x96_is_in_node_10 >= -1
 R2427: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.4 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x97_is_in_node_10 <= 1.1
 R2428: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_10 <= 1.1
 R2429: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x97_is_in_node_10 >= -1
 R2430: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 3.1 node_5_considers_var_2
   + 1.2 node_5_considers_var_3 + 1.1 x98_is_in_node_10 <= 1.1
 R2431: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_10 <= 1.1
 R2432: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x98_is_in_node_10 >= -1
 R2433: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.2 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x99_is_in_node_10 <= 1.1
 R2434: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_10 <= 1.1
 R2435: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x99_is_in_node_10 >= -1
 R2436: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 6.1 node_5_considers_var_2
   + 2.6 node_5_considers_var_3 + 1.1 x100_is_in_node_10 <= 1.1
 R2437: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_10 <= 1.1
 R2438: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x100_is_in_node_10 >= -1
 R2439: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x101_is_in_node_10 <= 1.1
 R2440: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_10 <= 1.1
 R2441: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x101_is_in_node_10 >= -1
 R2442: - split_value_node_5 + 7.2 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 6 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x102_is_in_node_10 <= 1.1
 R2443: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_10 <= 1.1
 R2444: - split_value_node_2 + 7.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x102_is_in_node_10 >= -1
 R2445: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x103_is_in_node_10 <= 1.1
 R2446: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_10 <= 1.1
 R2447: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x103_is_in_node_10 >= -1
 R2448: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.9 node_5_considers_var_2
   + 2.3 node_5_considers_var_3 + 1.1 x104_is_in_node_10 <= 1.1
 R2449: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_10 <= 1.1
 R2450: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x104_is_in_node_10 >= -1
 R2451: - split_value_node_5 + 7.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 6.7 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x105_is_in_node_10 <= 1.1
 R2452: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_10 <= 1.1
 R2453: - split_value_node_2 + 7.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x105_is_in_node_10 >= -1
 R2454: - split_value_node_5 + 5 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.8 node_5_considers_var_3 + 1.1 x106_is_in_node_10 <= 1.1
 R2455: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_10 <= 1.1
 R2456: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x106_is_in_node_10 >= -1
 R2457: - split_value_node_5 + 7.4 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 6.4 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x107_is_in_node_10 <= 1.1
 R2458: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_10 <= 1.1
 R2459: - split_value_node_2 + 7.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 6.3 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x107_is_in_node_10 >= -1
 R2460: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x108_is_in_node_10 <= 1.1
 R2461: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_10 <= 1.1
 R2462: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x108_is_in_node_10 >= -1
 R2463: - split_value_node_5 + 7.3 node_5_considers_var_0
   + 3.7 node_5_considers_var_1 + 6.2 node_5_considers_var_2
   + 2.6 node_5_considers_var_3 + 1.1 x109_is_in_node_10 <= 1.1
 R2464: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_10 <= 1.1
 R2465: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x109_is_in_node_10 >= -1
 R2466: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x110_is_in_node_10 <= 1.1
 R2467: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_10 <= 1.1
 R2468: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x110_is_in_node_10 >= -1
 R2469: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.4 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x111_is_in_node_10 <= 1.1
 R2470: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_10 <= 1.1
 R2471: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x111_is_in_node_10 >= -1
 R2472: - split_value_node_5 + 6.9 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.6 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x112_is_in_node_10 <= 1.1
 R2473: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_10 <= 1.1
 R2474: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x112_is_in_node_10 >= -1
 R2475: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x113_is_in_node_10 <= 1.1
 R2476: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_10 <= 1.1
 R2477: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x113_is_in_node_10 >= -1
 R2478: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2.5 node_5_considers_var_3 + 1.1 x114_is_in_node_10 <= 1.1
 R2479: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_10 <= 1.1
 R2480: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x114_is_in_node_10 >= -1
 R2481: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 5.4 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x115_is_in_node_10 <= 1.1
 R2482: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_10 <= 1.1
 R2483: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x115_is_in_node_10 >= -1
 R2484: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.6 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x116_is_in_node_10 <= 1.1
 R2485: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_10 <= 1.1
 R2486: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x116_is_in_node_10 >= -1
 R2487: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 6.8 node_5_considers_var_2
   + 2.3 node_5_considers_var_3 + 1.1 x117_is_in_node_10 <= 1.1
 R2488: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_10 <= 1.1
 R2489: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x117_is_in_node_10 >= -1
 R2490: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 7 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x118_is_in_node_10 <= 1.1
 R2491: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_10 <= 1.1
 R2492: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 6.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x118_is_in_node_10 >= -1
 R2493: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 2.3 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x119_is_in_node_10 <= 1.1
 R2494: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_10 <= 1.1
 R2495: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x119_is_in_node_10 >= -1
 R2496: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 5.8 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x120_is_in_node_10 <= 1.1
 R2497: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_10 <= 1.1
 R2498: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x120_is_in_node_10 >= -1
 R2499: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x121_is_in_node_10 <= 1.1
 R2500: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_10 <= 1.1
 R2501: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x121_is_in_node_10 >= -1
 R2502: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 6.8 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x122_is_in_node_10 <= 1.1
 R2503: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_10 <= 1.1
 R2504: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x122_is_in_node_10 >= -1
 R2505: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x123_is_in_node_10 <= 1.1
 R2506: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_10 <= 1.1
 R2507: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x123_is_in_node_10 >= -1
 R2508: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 5.8 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x124_is_in_node_10 <= 1.1
 R2509: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_10 <= 1.1
 R2510: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x124_is_in_node_10 >= -1
 R2511: - split_value_node_5 + 7.3 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 6.1 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x125_is_in_node_10 <= 1.1
 R2512: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_10 <= 1.1
 R2513: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x125_is_in_node_10 >= -1
 R2514: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x126_is_in_node_10 <= 1.1
 R2515: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_10 <= 1.1
 R2516: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x126_is_in_node_10 >= -1
 R2517: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x127_is_in_node_10 <= 1.1
 R2518: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_10 <= 1.1
 R2519: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x127_is_in_node_10 >= -1
 R2520: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x128_is_in_node_10 <= 1.1
 R2521: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_10 <= 1.1
 R2522: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x128_is_in_node_10 >= -1
 R2523: - split_value_node_5 + 7.3 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.9 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x129_is_in_node_10 <= 1.1
 R2524: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_10 <= 1.1
 R2525: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x129_is_in_node_10 >= -1
 R2526: - split_value_node_5 + 7.5 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 6.2 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x130_is_in_node_10 <= 1.1
 R2527: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_10 <= 1.1
 R2528: - split_value_node_2 + 7.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x130_is_in_node_10 >= -1
 R2529: - split_value_node_5 + 8 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 6.5 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x131_is_in_node_10 <= 1.1
 R2530: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_10 <= 1.1
 R2531: - split_value_node_2 + 7.9 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.4 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x131_is_in_node_10 >= -1
 R2532: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.3 node_5_considers_var_3 + 1.1 x132_is_in_node_10 <= 1.1
 R2533: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_10 <= 1.1
 R2534: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x132_is_in_node_10 >= -1
 R2535: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x133_is_in_node_10 <= 1.1
 R2536: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_10 <= 1.1
 R2537: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x133_is_in_node_10 >= -1
 R2538: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x134_is_in_node_10 <= 1.1
 R2539: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_10 <= 1.1
 R2540: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x134_is_in_node_10 >= -1
 R2541: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 6.2 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x135_is_in_node_10 <= 1.1
 R2542: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_10 <= 1.1
 R2543: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x135_is_in_node_10 >= -1
 R2544: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.5 node_5_considers_var_3 + 1.1 x136_is_in_node_10 <= 1.1
 R2545: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_10 <= 1.1
 R2546: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x136_is_in_node_10 >= -1
 R2547: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.6 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x137_is_in_node_10 <= 1.1
 R2548: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_10 <= 1.1
 R2549: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x137_is_in_node_10 >= -1
 R2550: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x138_is_in_node_10 <= 1.1
 R2551: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_10 <= 1.1
 R2552: - split_value_node_2 + 6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x138_is_in_node_10 >= -1
 R2553: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.5 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x139_is_in_node_10 <= 1.1
 R2554: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_10 <= 1.1
 R2555: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x139_is_in_node_10 >= -1
 R2556: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.5 node_5_considers_var_3 + 1.1 x140_is_in_node_10 <= 1.1
 R2557: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_10 <= 1.1
 R2558: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x140_is_in_node_10 >= -1
 R2559: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x141_is_in_node_10 <= 1.1
 R2560: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_10 <= 1.1
 R2561: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x141_is_in_node_10 >= -1
 R2562: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x142_is_in_node_10 <= 1.1
 R2563: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_10 <= 1.1
 R2564: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x142_is_in_node_10 >= -1
 R2565: - split_value_node_5 + 6.9 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 6 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x143_is_in_node_10 <= 1.1
 R2566: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_10 <= 1.1
 R2567: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x143_is_in_node_10 >= -1
 R2568: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 5.8 node_5_considers_var_2
   + 2.6 node_5_considers_var_3 + 1.1 x144_is_in_node_10 <= 1.1
 R2569: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_10 <= 1.1
 R2570: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x144_is_in_node_10 >= -1
 R2571: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.3 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x145_is_in_node_10 <= 1.1
 R2572: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_10 <= 1.1
 R2573: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x145_is_in_node_10 >= -1
 R2574: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x146_is_in_node_10 <= 1.1
 R2575: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_10 <= 1.1
 R2576: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x146_is_in_node_10 >= -1
 R2577: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.3 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x147_is_in_node_10 <= 1.1
 R2578: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_10 <= 1.1
 R2579: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x147_is_in_node_10 >= -1
 R2580: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 5.5 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x148_is_in_node_10 <= 1.1
 R2581: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_10 <= 1.1
 R2582: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x148_is_in_node_10 >= -1
 R2583: - split_value_node_5 + 6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x149_is_in_node_10 <= 1.1
 R2584: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_10 <= 1.1
 R2585: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x149_is_in_node_10 >= -1
 R2586: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x0_is_in_node_11 <= 1.1
 R2587: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_11 <= 1.1
 R2588: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x0_is_in_node_11 >= -1
 R2589: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x1_is_in_node_11 <= 1.1
 R2590: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_11 <= 1.1
 R2591: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x1_is_in_node_11 >= -1
 R2592: - split_value_node_5 + 4.8 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x2_is_in_node_11 <= 1.1
 R2593: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_11 <= 1.1
 R2594: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x2_is_in_node_11 >= -1
 R2595: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x3_is_in_node_11 <= 1.1
 R2596: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_11 <= 1.1
 R2597: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x3_is_in_node_11 >= -1
 R2598: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.7 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x4_is_in_node_11 <= 1.1
 R2599: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_11 <= 1.1
 R2600: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x4_is_in_node_11 >= -1
 R2601: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 4 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x5_is_in_node_11 <= 1.1
 R2602: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_11 <= 1.1
 R2603: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x5_is_in_node_11 >= -1
 R2604: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x6_is_in_node_11 <= 1.1
 R2605: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_11 <= 1.1
 R2606: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x6_is_in_node_11 >= -1
 R2607: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x7_is_in_node_11 <= 1.1
 R2608: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_11 <= 1.1
 R2609: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x7_is_in_node_11 >= -1
 R2610: - split_value_node_5 + 4.5 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x8_is_in_node_11 <= 1.1
 R2611: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_11 <= 1.1
 R2612: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x8_is_in_node_11 >= -1
 R2613: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x9_is_in_node_11 <= 1.1
 R2614: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_11 <= 1.1
 R2615: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x9_is_in_node_11 >= -1
 R2616: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.8 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x10_is_in_node_11 <= 1.1
 R2617: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_11 <= 1.1
 R2618: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x10_is_in_node_11 >= -1
 R2619: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x11_is_in_node_11 <= 1.1
 R2620: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_11 <= 1.1
 R2621: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x11_is_in_node_11 >= -1
 R2622: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x12_is_in_node_11 <= 1.1
 R2623: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_11 <= 1.1
 R2624: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x12_is_in_node_11 >= -1
 R2625: - split_value_node_5 + 4.4 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.2 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x13_is_in_node_11 <= 1.1
 R2626: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_11 <= 1.1
 R2627: - split_value_node_2 + 4.3 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.1 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x13_is_in_node_11 >= -1
 R2628: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 4.1 node_5_considers_var_1 + 1.3 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x14_is_in_node_11 <= 1.1
 R2629: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_11 <= 1.1
 R2630: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 4 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x14_is_in_node_11 >= -1
 R2631: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 4.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x15_is_in_node_11 <= 1.1
 R2632: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_11 <= 1.1
 R2633: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 4.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x15_is_in_node_11 >= -1
 R2634: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 4 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x16_is_in_node_11 <= 1.1
 R2635: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_11 <= 1.1
 R2636: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.9 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x16_is_in_node_11 >= -1
 R2637: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x17_is_in_node_11 <= 1.1
 R2638: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_11 <= 1.1
 R2639: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x17_is_in_node_11 >= -1
 R2640: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x18_is_in_node_11 <= 1.1
 R2641: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_11 <= 1.1
 R2642: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x18_is_in_node_11 >= -1
 R2643: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x19_is_in_node_11 <= 1.1
 R2644: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_11 <= 1.1
 R2645: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x19_is_in_node_11 >= -1
 R2646: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x20_is_in_node_11 <= 1.1
 R2647: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_11 <= 1.1
 R2648: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x20_is_in_node_11 >= -1
 R2649: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.8 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x21_is_in_node_11 <= 1.1
 R2650: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_11 <= 1.1
 R2651: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x21_is_in_node_11 >= -1
 R2652: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.7 node_5_considers_var_1 + 1.1 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x22_is_in_node_11 <= 1.1
 R2653: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_11 <= 1.1
 R2654: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x22_is_in_node_11 >= -1
 R2655: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 1.8 node_5_considers_var_2
   + 0.6 node_5_considers_var_3 + 1.1 x23_is_in_node_11 <= 1.1
 R2656: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_11 <= 1.1
 R2657: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.7 node_2_considers_var_2
   + 0.5 node_2_considers_var_3 - x23_is_in_node_11 >= -1
 R2658: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 2 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x24_is_in_node_11 <= 1.1
 R2659: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_11 <= 1.1
 R2660: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x24_is_in_node_11 >= -1
 R2661: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x25_is_in_node_11 <= 1.1
 R2662: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_11 <= 1.1
 R2663: - split_value_node_2 + 5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x25_is_in_node_11 >= -1
 R2664: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x26_is_in_node_11 <= 1.1
 R2665: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_11 <= 1.1
 R2666: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x26_is_in_node_11 >= -1
 R2667: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x27_is_in_node_11 <= 1.1
 R2668: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_11 <= 1.1
 R2669: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x27_is_in_node_11 >= -1
 R2670: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x28_is_in_node_11 <= 1.1
 R2671: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_11 <= 1.1
 R2672: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x28_is_in_node_11 >= -1
 R2673: - split_value_node_5 + 4.8 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x29_is_in_node_11 <= 1.1
 R2674: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_11 <= 1.1
 R2675: - split_value_node_2 + 4.7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x29_is_in_node_11 >= -1
 R2676: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x30_is_in_node_11 <= 1.1
 R2677: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_11 <= 1.1
 R2678: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x30_is_in_node_11 >= -1
 R2679: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x31_is_in_node_11 <= 1.1
 R2680: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_11 <= 1.1
 R2681: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x31_is_in_node_11 >= -1
 R2682: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 4.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x32_is_in_node_11 <= 1.1
 R2683: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_11 <= 1.1
 R2684: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 4.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x32_is_in_node_11 >= -1
 R2685: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 4.3 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x33_is_in_node_11 <= 1.1
 R2686: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_11 <= 1.1
 R2687: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 4.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x33_is_in_node_11 >= -1
 R2688: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x34_is_in_node_11 <= 1.1
 R2689: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_11 <= 1.1
 R2690: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x34_is_in_node_11 >= -1
 R2691: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.3 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x35_is_in_node_11 <= 1.1
 R2692: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_11 <= 1.1
 R2693: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.2 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x35_is_in_node_11 >= -1
 R2694: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x36_is_in_node_11 <= 1.1
 R2695: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_11 <= 1.1
 R2696: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x36_is_in_node_11 >= -1
 R2697: - split_value_node_5 + 5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.2 node_5_considers_var_3 + 1.1 x37_is_in_node_11 <= 1.1
 R2698: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_11 <= 1.1
 R2699: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.1 node_2_considers_var_3 - x37_is_in_node_11 >= -1
 R2700: - split_value_node_5 + 4.5 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x38_is_in_node_11 <= 1.1
 R2701: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_11 <= 1.1
 R2702: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x38_is_in_node_11 >= -1
 R2703: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x39_is_in_node_11 <= 1.1
 R2704: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_11 <= 1.1
 R2705: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x39_is_in_node_11 >= -1
 R2706: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x40_is_in_node_11 <= 1.1
 R2707: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_11 <= 1.1
 R2708: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x40_is_in_node_11 >= -1
 R2709: - split_value_node_5 + 4.6 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x41_is_in_node_11 <= 1.1
 R2710: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_11 <= 1.1
 R2711: - split_value_node_2 + 4.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x41_is_in_node_11 >= -1
 R2712: - split_value_node_5 + 4.5 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.4 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x42_is_in_node_11 <= 1.1
 R2713: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_11 <= 1.1
 R2714: - split_value_node_2 + 4.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.3 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x42_is_in_node_11 >= -1
 R2715: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.6 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.7 node_5_considers_var_3 + 1.1 x43_is_in_node_11 <= 1.1
 R2716: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_11 <= 1.1
 R2717: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.5 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.6 node_2_considers_var_3 - x43_is_in_node_11 >= -1
 R2718: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 2 node_5_considers_var_2
   + 0.5 node_5_considers_var_3 + 1.1 x44_is_in_node_11 <= 1.1
 R2719: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_11 <= 1.1
 R2720: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.9 node_2_considers_var_2
   + 0.4 node_2_considers_var_3 - x44_is_in_node_11 >= -1
 R2721: - split_value_node_5 + 4.9 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.4 node_5_considers_var_3 + 1.1 x45_is_in_node_11 <= 1.1
 R2722: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_11 <= 1.1
 R2723: - split_value_node_2 + 4.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.3 node_2_considers_var_3 - x45_is_in_node_11 >= -1
 R2724: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 1.7 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x46_is_in_node_11 <= 1.1
 R2725: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_11 <= 1.1
 R2726: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 1.6 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x46_is_in_node_11 >= -1
 R2727: - split_value_node_5 + 4.7 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x47_is_in_node_11 <= 1.1
 R2728: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_11 <= 1.1
 R2729: - split_value_node_2 + 4.6 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x47_is_in_node_11 >= -1
 R2730: - split_value_node_5 + 5.4 node_5_considers_var_0
   + 3.8 node_5_considers_var_1 + 1.6 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x48_is_in_node_11 <= 1.1
 R2731: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_11 <= 1.1
 R2732: - split_value_node_2 + 5.3 node_2_considers_var_0
   + 3.7 node_2_considers_var_1 + 1.5 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x48_is_in_node_11 >= -1
 R2733: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 1.5 node_5_considers_var_2
   + 0.3 node_5_considers_var_3 + 1.1 x49_is_in_node_11 <= 1.1
 R2734: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_11 <= 1.1
 R2735: - split_value_node_2 + 5 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 1.4 node_2_considers_var_2
   + 0.2 node_2_considers_var_3 - x49_is_in_node_11 >= -1
 R2736: - split_value_node_5 + 7.1 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x50_is_in_node_11 <= 1.1
 R2737: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_11 <= 1.1
 R2738: - split_value_node_2 + 7 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x50_is_in_node_11 >= -1
 R2739: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x51_is_in_node_11 <= 1.1
 R2740: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_11 <= 1.1
 R2741: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x51_is_in_node_11 >= -1
 R2742: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x52_is_in_node_11 <= 1.1
 R2743: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_11 <= 1.1
 R2744: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x52_is_in_node_11 >= -1
 R2745: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x53_is_in_node_11 <= 1.1
 R2746: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_11 <= 1.1
 R2747: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x53_is_in_node_11 >= -1
 R2748: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.7 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x54_is_in_node_11 <= 1.1
 R2749: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_11 <= 1.1
 R2750: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x54_is_in_node_11 >= -1
 R2751: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x55_is_in_node_11 <= 1.1
 R2752: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_11 <= 1.1
 R2753: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x55_is_in_node_11 >= -1
 R2754: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x56_is_in_node_11 <= 1.1
 R2755: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_11 <= 1.1
 R2756: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x56_is_in_node_11 >= -1
 R2757: - split_value_node_5 + 5 node_5_considers_var_0
   + 2.5 node_5_considers_var_1 + 3.4 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x57_is_in_node_11 <= 1.1
 R2758: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_11 <= 1.1
 R2759: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x57_is_in_node_11 >= -1
 R2760: - split_value_node_5 + 6.7 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.7 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x58_is_in_node_11 <= 1.1
 R2761: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_11 <= 1.1
 R2762: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x58_is_in_node_11 >= -1
 R2763: - split_value_node_5 + 5.3 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x59_is_in_node_11 <= 1.1
 R2764: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_11 <= 1.1
 R2765: - split_value_node_2 + 5.2 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x59_is_in_node_11 >= -1
 R2766: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 2.1 node_5_considers_var_1 + 3.6 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x60_is_in_node_11 <= 1.1
 R2767: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_11 <= 1.1
 R2768: - split_value_node_2 + 5 node_2_considers_var_0
   + 2 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x60_is_in_node_11 >= -1
 R2769: - split_value_node_5 + 6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x61_is_in_node_11 <= 1.1
 R2770: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_11 <= 1.1
 R2771: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x61_is_in_node_11 >= -1
 R2772: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 2.3 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x62_is_in_node_11 <= 1.1
 R2773: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_11 <= 1.1
 R2774: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4 node_2_considers_var_2
   + node_2_considers_var_3 - x62_is_in_node_11 >= -1
 R2775: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x63_is_in_node_11 <= 1.1
 R2776: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_11 <= 1.1
 R2777: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x63_is_in_node_11 >= -1
 R2778: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 3.7 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x64_is_in_node_11 <= 1.1
 R2779: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_11 <= 1.1
 R2780: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 3.6 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x64_is_in_node_11 >= -1
 R2781: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x65_is_in_node_11 <= 1.1
 R2782: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_11 <= 1.1
 R2783: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x65_is_in_node_11 >= -1
 R2784: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x66_is_in_node_11 <= 1.1
 R2785: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_11 <= 1.1
 R2786: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x66_is_in_node_11 >= -1
 R2787: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4.2 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x67_is_in_node_11 <= 1.1
 R2788: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_11 <= 1.1
 R2789: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + node_2_considers_var_3 - x67_is_in_node_11 >= -1
 R2790: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 2.3 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x68_is_in_node_11 <= 1.1
 R2791: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_11 <= 1.1
 R2792: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x68_is_in_node_11 >= -1
 R2793: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 4 node_5_considers_var_2
   + 1.2 node_5_considers_var_3 + 1.1 x69_is_in_node_11 <= 1.1
 R2794: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_11 <= 1.1
 R2795: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x69_is_in_node_11 >= -1
 R2796: - split_value_node_5 + 6 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x70_is_in_node_11 <= 1.1
 R2797: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_11 <= 1.1
 R2798: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x70_is_in_node_11 >= -1
 R2799: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x71_is_in_node_11 <= 1.1
 R2800: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_11 <= 1.1
 R2801: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x71_is_in_node_11 >= -1
 R2802: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x72_is_in_node_11 <= 1.1
 R2803: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_11 <= 1.1
 R2804: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x72_is_in_node_11 >= -1
 R2805: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x73_is_in_node_11 <= 1.1
 R2806: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_11 <= 1.1
 R2807: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x73_is_in_node_11 >= -1
 R2808: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.4 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x74_is_in_node_11 <= 1.1
 R2809: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_11 <= 1.1
 R2810: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x74_is_in_node_11 >= -1
 R2811: - split_value_node_5 + 6.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x75_is_in_node_11 <= 1.1
 R2812: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_11 <= 1.1
 R2813: - split_value_node_2 + 6.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x75_is_in_node_11 >= -1
 R2814: - split_value_node_5 + 6.9 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x76_is_in_node_11 <= 1.1
 R2815: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_11 <= 1.1
 R2816: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x76_is_in_node_11 >= -1
 R2817: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 1.8 node_5_considers_var_3 + 1.1 x77_is_in_node_11 <= 1.1
 R2818: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_11 <= 1.1
 R2819: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x77_is_in_node_11 >= -1
 R2820: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x78_is_in_node_11 <= 1.1
 R2821: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_11 <= 1.1
 R2822: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x78_is_in_node_11 >= -1
 R2823: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 3.6 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x79_is_in_node_11 <= 1.1
 R2824: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_11 <= 1.1
 R2825: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 3.5 node_2_considers_var_2
   + node_2_considers_var_3 - x79_is_in_node_11 >= -1
 R2826: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.5 node_5_considers_var_1 + 3.9 node_5_considers_var_2
   + 1.2 node_5_considers_var_3 + 1.1 x80_is_in_node_11 <= 1.1
 R2827: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_11 <= 1.1
 R2828: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.8 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x80_is_in_node_11 >= -1
 R2829: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.5 node_5_considers_var_1 + 3.8 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x81_is_in_node_11 <= 1.1
 R2830: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_11 <= 1.1
 R2831: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.4 node_2_considers_var_1 + 3.7 node_2_considers_var_2
   + node_2_considers_var_3 - x81_is_in_node_11 >= -1
 R2832: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x82_is_in_node_11 <= 1.1
 R2833: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_11 <= 1.1
 R2834: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 3.9 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x82_is_in_node_11 >= -1
 R2835: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x83_is_in_node_11 <= 1.1
 R2836: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_11 <= 1.1
 R2837: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x83_is_in_node_11 >= -1
 R2838: - split_value_node_5 + 5.5 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x84_is_in_node_11 <= 1.1
 R2839: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_11 <= 1.1
 R2840: - split_value_node_2 + 5.4 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x84_is_in_node_11 >= -1
 R2841: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x85_is_in_node_11 <= 1.1
 R2842: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_11 <= 1.1
 R2843: - split_value_node_2 + 6 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x85_is_in_node_11 >= -1
 R2844: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 4.8 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x86_is_in_node_11 <= 1.1
 R2845: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_11 <= 1.1
 R2846: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 4.7 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x86_is_in_node_11 >= -1
 R2847: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x87_is_in_node_11 <= 1.1
 R2848: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_11 <= 1.1
 R2849: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x87_is_in_node_11 >= -1
 R2850: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.2 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x88_is_in_node_11 <= 1.1
 R2851: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_11 <= 1.1
 R2852: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x88_is_in_node_11 >= -1
 R2853: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x89_is_in_node_11 <= 1.1
 R2854: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_11 <= 1.1
 R2855: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x89_is_in_node_11 >= -1
 R2856: - split_value_node_5 + 5.6 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 4.5 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x90_is_in_node_11 <= 1.1
 R2857: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_11 <= 1.1
 R2858: - split_value_node_2 + 5.5 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4.4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x90_is_in_node_11 >= -1
 R2859: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.7 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x91_is_in_node_11 <= 1.1
 R2860: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_11 <= 1.1
 R2861: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x91_is_in_node_11 >= -1
 R2862: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 4.1 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x92_is_in_node_11 <= 1.1
 R2863: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_11 <= 1.1
 R2864: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 4 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x92_is_in_node_11 >= -1
 R2865: - split_value_node_5 + 5.1 node_5_considers_var_0
   + 2.4 node_5_considers_var_1 + 3.4 node_5_considers_var_2
   + 1.1 node_5_considers_var_3 + 1.1 x93_is_in_node_11 <= 1.1
 R2866: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_11 <= 1.1
 R2867: - split_value_node_2 + 5 node_2_considers_var_0
   + 2.3 node_2_considers_var_1 + 3.3 node_2_considers_var_2
   + node_2_considers_var_3 - x93_is_in_node_11 >= -1
 R2868: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x94_is_in_node_11 <= 1.1
 R2869: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_11 <= 1.1
 R2870: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x94_is_in_node_11 >= -1
 R2871: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.3 node_5_considers_var_3 + 1.1 x95_is_in_node_11 <= 1.1
 R2872: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_11 <= 1.1
 R2873: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.2 node_2_considers_var_3 - x95_is_in_node_11 >= -1
 R2874: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.3 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x96_is_in_node_11 <= 1.1
 R2875: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_11 <= 1.1
 R2876: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.2 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x96_is_in_node_11 >= -1
 R2877: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 4.4 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x97_is_in_node_11 <= 1.1
 R2878: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_11 <= 1.1
 R2879: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 4.3 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x97_is_in_node_11 >= -1
 R2880: - split_value_node_5 + 5.2 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 3.1 node_5_considers_var_2
   + 1.2 node_5_considers_var_3 + 1.1 x98_is_in_node_11 <= 1.1
 R2881: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_11 <= 1.1
 R2882: - split_value_node_2 + 5.1 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 3 node_2_considers_var_2
   + 1.1 node_2_considers_var_3 - x98_is_in_node_11 >= -1
 R2883: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.2 node_5_considers_var_2
   + 1.4 node_5_considers_var_3 + 1.1 x99_is_in_node_11 <= 1.1
 R2884: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_11 <= 1.1
 R2885: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.1 node_2_considers_var_2
   + 1.3 node_2_considers_var_3 - x99_is_in_node_11 >= -1
 R2886: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 6.1 node_5_considers_var_2
   + 2.6 node_5_considers_var_3 + 1.1 x100_is_in_node_11 <= 1.1
 R2887: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_11 <= 1.1
 R2888: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x100_is_in_node_11 >= -1
 R2889: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x101_is_in_node_11 <= 1.1
 R2890: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_11 <= 1.1
 R2891: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x101_is_in_node_11 >= -1
 R2892: - split_value_node_5 + 7.2 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 6 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x102_is_in_node_11 <= 1.1
 R2893: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_11 <= 1.1
 R2894: - split_value_node_2 + 7.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x102_is_in_node_11 >= -1
 R2895: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x103_is_in_node_11 <= 1.1
 R2896: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_11 <= 1.1
 R2897: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x103_is_in_node_11 >= -1
 R2898: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.9 node_5_considers_var_2
   + 2.3 node_5_considers_var_3 + 1.1 x104_is_in_node_11 <= 1.1
 R2899: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_11 <= 1.1
 R2900: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x104_is_in_node_11 >= -1
 R2901: - split_value_node_5 + 7.7 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 6.7 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x105_is_in_node_11 <= 1.1
 R2902: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_11 <= 1.1
 R2903: - split_value_node_2 + 7.6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x105_is_in_node_11 >= -1
 R2904: - split_value_node_5 + 5 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 4.6 node_5_considers_var_2
   + 1.8 node_5_considers_var_3 + 1.1 x106_is_in_node_11 <= 1.1
 R2905: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_11 <= 1.1
 R2906: - split_value_node_2 + 4.9 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 4.5 node_2_considers_var_2
   + 1.7 node_2_considers_var_3 - x106_is_in_node_11 >= -1
 R2907: - split_value_node_5 + 7.4 node_5_considers_var_0
   + 3 node_5_considers_var_1 + 6.4 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x107_is_in_node_11 <= 1.1
 R2908: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_11 <= 1.1
 R2909: - split_value_node_2 + 7.3 node_2_considers_var_0
   + 2.9 node_2_considers_var_1 + 6.3 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x107_is_in_node_11 >= -1
 R2910: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x108_is_in_node_11 <= 1.1
 R2911: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_11 <= 1.1
 R2912: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x108_is_in_node_11 >= -1
 R2913: - split_value_node_5 + 7.3 node_5_considers_var_0
   + 3.7 node_5_considers_var_1 + 6.2 node_5_considers_var_2
   + 2.6 node_5_considers_var_3 + 1.1 x109_is_in_node_11 <= 1.1
 R2914: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_11 <= 1.1
 R2915: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.6 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x109_is_in_node_11 >= -1
 R2916: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x110_is_in_node_11 <= 1.1
 R2917: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_11 <= 1.1
 R2918: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x110_is_in_node_11 >= -1
 R2919: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.4 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x111_is_in_node_11 <= 1.1
 R2920: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_11 <= 1.1
 R2921: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x111_is_in_node_11 >= -1
 R2922: - split_value_node_5 + 6.9 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.6 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x112_is_in_node_11 <= 1.1
 R2923: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_11 <= 1.1
 R2924: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x112_is_in_node_11 >= -1
 R2925: - split_value_node_5 + 5.8 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x113_is_in_node_11 <= 1.1
 R2926: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_11 <= 1.1
 R2927: - split_value_node_2 + 5.7 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x113_is_in_node_11 >= -1
 R2928: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2.5 node_5_considers_var_3 + 1.1 x114_is_in_node_11 <= 1.1
 R2929: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_11 <= 1.1
 R2930: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x114_is_in_node_11 >= -1
 R2931: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 5.4 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x115_is_in_node_11 <= 1.1
 R2932: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_11 <= 1.1
 R2933: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.3 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x115_is_in_node_11 >= -1
 R2934: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.6 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x116_is_in_node_11 <= 1.1
 R2935: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_11 <= 1.1
 R2936: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x116_is_in_node_11 >= -1
 R2937: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 6.8 node_5_considers_var_2
   + 2.3 node_5_considers_var_3 + 1.1 x117_is_in_node_11 <= 1.1
 R2938: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_11 <= 1.1
 R2939: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x117_is_in_node_11 >= -1
 R2940: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 7 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x118_is_in_node_11 <= 1.1
 R2941: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_11 <= 1.1
 R2942: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 6.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x118_is_in_node_11 >= -1
 R2943: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 2.3 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x119_is_in_node_11 <= 1.1
 R2944: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_11 <= 1.1
 R2945: - split_value_node_2 + 6 node_2_considers_var_0
   + 2.2 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x119_is_in_node_11 >= -1
 R2946: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 5.8 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x120_is_in_node_11 <= 1.1
 R2947: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_11 <= 1.1
 R2948: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x120_is_in_node_11 >= -1
 R2949: - split_value_node_5 + 5.7 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x121_is_in_node_11 <= 1.1
 R2950: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_11 <= 1.1
 R2951: - split_value_node_2 + 5.6 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x121_is_in_node_11 >= -1
 R2952: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 6.8 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x122_is_in_node_11 <= 1.1
 R2953: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_11 <= 1.1
 R2954: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.7 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x122_is_in_node_11 >= -1
 R2955: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x123_is_in_node_11 <= 1.1
 R2956: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_11 <= 1.1
 R2957: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x123_is_in_node_11 >= -1
 R2958: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 5.8 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x124_is_in_node_11 <= 1.1
 R2959: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_11 <= 1.1
 R2960: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x124_is_in_node_11 >= -1
 R2961: - split_value_node_5 + 7.3 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 6.1 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x125_is_in_node_11 <= 1.1
 R2962: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_11 <= 1.1
 R2963: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 6 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x125_is_in_node_11 >= -1
 R2964: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x126_is_in_node_11 <= 1.1
 R2965: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_11 <= 1.1
 R2966: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x126_is_in_node_11 >= -1
 R2967: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x127_is_in_node_11 <= 1.1
 R2968: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_11 <= 1.1
 R2969: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.9 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x127_is_in_node_11 >= -1
 R2970: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x128_is_in_node_11 <= 1.1
 R2971: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_11 <= 1.1
 R2972: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x128_is_in_node_11 >= -1
 R2973: - split_value_node_5 + 7.3 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.9 node_5_considers_var_2
   + 1.7 node_5_considers_var_3 + 1.1 x129_is_in_node_11 <= 1.1
 R2974: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_11 <= 1.1
 R2975: - split_value_node_2 + 7.2 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.8 node_2_considers_var_2
   + 1.6 node_2_considers_var_3 - x129_is_in_node_11 >= -1
 R2976: - split_value_node_5 + 7.5 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 6.2 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x130_is_in_node_11 <= 1.1
 R2977: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_11 <= 1.1
 R2978: - split_value_node_2 + 7.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x130_is_in_node_11 >= -1
 R2979: - split_value_node_5 + 8 node_5_considers_var_0
   + 3.9 node_5_considers_var_1 + 6.5 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x131_is_in_node_11 <= 1.1
 R2980: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_11 <= 1.1
 R2981: - split_value_node_2 + 7.9 node_2_considers_var_0
   + 3.8 node_2_considers_var_1 + 6.4 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x131_is_in_node_11 >= -1
 R2982: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.3 node_5_considers_var_3 + 1.1 x132_is_in_node_11 <= 1.1
 R2983: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_11 <= 1.1
 R2984: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.2 node_2_considers_var_3 - x132_is_in_node_11 >= -1
 R2985: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.9 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 1.6 node_5_considers_var_3 + 1.1 x133_is_in_node_11 <= 1.1
 R2986: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_11 <= 1.1
 R2987: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.8 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.5 node_2_considers_var_3 - x133_is_in_node_11 >= -1
 R2988: - split_value_node_5 + 6.2 node_5_considers_var_0
   + 2.7 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 1.5 node_5_considers_var_3 + 1.1 x134_is_in_node_11 <= 1.1
 R2989: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_11 <= 1.1
 R2990: - split_value_node_2 + 6.1 node_2_considers_var_0
   + 2.6 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 1.4 node_2_considers_var_3 - x134_is_in_node_11 >= -1
 R2991: - split_value_node_5 + 7.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 6.2 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x135_is_in_node_11 <= 1.1
 R2992: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_11 <= 1.1
 R2993: - split_value_node_2 + 7.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 6.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x135_is_in_node_11 >= -1
 R2994: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.5 node_5_considers_var_3 + 1.1 x136_is_in_node_11 <= 1.1
 R2995: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_11 <= 1.1
 R2996: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x136_is_in_node_11 >= -1
 R2997: - split_value_node_5 + 6.5 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.6 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x137_is_in_node_11 <= 1.1
 R2998: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_11 <= 1.1
 R2999: - split_value_node_2 + 6.4 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.5 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x137_is_in_node_11 >= -1
 R3000: - split_value_node_5 + 6.1 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 4.9 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x138_is_in_node_11 <= 1.1
 R3001: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_11 <= 1.1
 R3002: - split_value_node_2 + 6 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 4.8 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x138_is_in_node_11 >= -1
 R3003: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.5 node_5_considers_var_2
   + 2.2 node_5_considers_var_3 + 1.1 x139_is_in_node_11 <= 1.1
 R3004: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_11 <= 1.1
 R3005: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.1 node_2_considers_var_3 - x139_is_in_node_11 >= -1
 R3006: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.7 node_5_considers_var_2
   + 2.5 node_5_considers_var_3 + 1.1 x140_is_in_node_11 <= 1.1
 R3007: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_11 <= 1.1
 R3008: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.6 node_2_considers_var_2
   + 2.4 node_2_considers_var_3 - x140_is_in_node_11 >= -1
 R3009: - split_value_node_5 + 7 node_5_considers_var_0
   + 3.2 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x141_is_in_node_11 <= 1.1
 R3010: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_11 <= 1.1
 R3011: - split_value_node_2 + 6.9 node_2_considers_var_0
   + 3.1 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x141_is_in_node_11 >= -1
 R3012: - split_value_node_5 + 5.9 node_5_considers_var_0
   + 2.8 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x142_is_in_node_11 <= 1.1
 R3013: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_11 <= 1.1
 R3014: - split_value_node_2 + 5.8 node_2_considers_var_0
   + 2.7 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x142_is_in_node_11 >= -1
 R3015: - split_value_node_5 + 6.9 node_5_considers_var_0
   + 3.3 node_5_considers_var_1 + 6 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x143_is_in_node_11 <= 1.1
 R3016: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_11 <= 1.1
 R3017: - split_value_node_2 + 6.8 node_2_considers_var_0
   + 3.2 node_2_considers_var_1 + 5.9 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x143_is_in_node_11 >= -1
 R3018: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.4 node_5_considers_var_1 + 5.8 node_5_considers_var_2
   + 2.6 node_5_considers_var_3 + 1.1 x144_is_in_node_11 <= 1.1
 R3019: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_11 <= 1.1
 R3020: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3.3 node_2_considers_var_1 + 5.7 node_2_considers_var_2
   + 2.5 node_2_considers_var_3 - x144_is_in_node_11 >= -1
 R3021: - split_value_node_5 + 6.8 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.3 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x145_is_in_node_11 <= 1.1
 R3022: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_11 <= 1.1
 R3023: - split_value_node_2 + 6.7 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x145_is_in_node_11 >= -1
 R3024: - split_value_node_5 + 6.4 node_5_considers_var_0
   + 2.6 node_5_considers_var_1 + 5.1 node_5_considers_var_2
   + 2 node_5_considers_var_3 + 1.1 x146_is_in_node_11 <= 1.1
 R3025: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_11 <= 1.1
 R3026: - split_value_node_2 + 6.3 node_2_considers_var_0
   + 2.5 node_2_considers_var_1 + 5 node_2_considers_var_2
   + 1.9 node_2_considers_var_3 - x146_is_in_node_11 >= -1
 R3027: - split_value_node_5 + 6.6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.3 node_5_considers_var_2
   + 2.1 node_5_considers_var_3 + 1.1 x147_is_in_node_11 <= 1.1
 R3028: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_11 <= 1.1
 R3029: - split_value_node_2 + 6.5 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.2 node_2_considers_var_2
   + 2 node_2_considers_var_3 - x147_is_in_node_11 >= -1
 R3030: - split_value_node_5 + 6.3 node_5_considers_var_0
   + 3.5 node_5_considers_var_1 + 5.5 node_5_considers_var_2
   + 2.4 node_5_considers_var_3 + 1.1 x148_is_in_node_11 <= 1.1
 R3031: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_11 <= 1.1
 R3032: - split_value_node_2 + 6.2 node_2_considers_var_0
   + 3.4 node_2_considers_var_1 + 5.4 node_2_considers_var_2
   + 2.3 node_2_considers_var_3 - x148_is_in_node_11 >= -1
 R3033: - split_value_node_5 + 6 node_5_considers_var_0
   + 3.1 node_5_considers_var_1 + 5.2 node_5_considers_var_2
   + 1.9 node_5_considers_var_3 + 1.1 x149_is_in_node_11 <= 1.1
 R3034: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_11 <= 1.1
 R3035: - split_value_node_2 + 5.9 node_2_considers_var_0
   + 3 node_2_considers_var_1 + 5.1 node_2_considers_var_2
   + 1.8 node_2_considers_var_3 - x149_is_in_node_11 >= -1
 R3036: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x0_is_in_node_12 <= 1.1
 R3037: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_12 <= 1.1
 R3038: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x0_is_in_node_12 >= -1
 R3039: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x1_is_in_node_12 <= 1.1
 R3040: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_12 <= 1.1
 R3041: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x1_is_in_node_12 >= -1
 R3042: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x2_is_in_node_12 <= 1.1
 R3043: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_12 <= 1.1
 R3044: - split_value_node_6 + 4.7 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x2_is_in_node_12 >= -1
 R3045: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x3_is_in_node_12 <= 1.1
 R3046: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_12 <= 1.1
 R3047: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x3_is_in_node_12 >= -1
 R3048: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x4_is_in_node_12 <= 1.1
 R3049: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_12 <= 1.1
 R3050: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.6 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x4_is_in_node_12 >= -1
 R3051: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x5_is_in_node_12 <= 1.1
 R3052: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_12 <= 1.1
 R3053: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.9 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x5_is_in_node_12 >= -1
 R3054: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x6_is_in_node_12 <= 1.1
 R3055: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_12 <= 1.1
 R3056: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x6_is_in_node_12 >= -1
 R3057: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x7_is_in_node_12 <= 1.1
 R3058: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_12 <= 1.1
 R3059: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x7_is_in_node_12 >= -1
 R3060: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x8_is_in_node_12 <= 1.1
 R3061: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_12 <= 1.1
 R3062: - split_value_node_6 + 4.4 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x8_is_in_node_12 >= -1
 R3063: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x9_is_in_node_12 <= 1.1
 R3064: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_12 <= 1.1
 R3065: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x9_is_in_node_12 >= -1
 R3066: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x10_is_in_node_12 <= 1.1
 R3067: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_12 <= 1.1
 R3068: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.7 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x10_is_in_node_12 >= -1
 R3069: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x11_is_in_node_12 <= 1.1
 R3070: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_12 <= 1.1
 R3071: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x11_is_in_node_12 >= -1
 R3072: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x12_is_in_node_12 <= 1.1
 R3073: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_12 <= 1.1
 R3074: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x12_is_in_node_12 >= -1
 R3075: - split_value_node_3 + 4.4 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.2 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x13_is_in_node_12 <= 1.1
 R3076: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_12 <= 1.1
 R3077: - split_value_node_6 + 4.3 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.1 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x13_is_in_node_12 >= -1
 R3078: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 4.1 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x14_is_in_node_12 <= 1.1
 R3079: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_12 <= 1.1
 R3080: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 4 node_6_considers_var_1 + 1.2 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x14_is_in_node_12 >= -1
 R3081: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 4.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x15_is_in_node_12 <= 1.1
 R3082: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_12 <= 1.1
 R3083: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 4.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x15_is_in_node_12 >= -1
 R3084: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x16_is_in_node_12 <= 1.1
 R3085: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_12 <= 1.1
 R3086: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.9 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x16_is_in_node_12 >= -1
 R3087: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x17_is_in_node_12 <= 1.1
 R3088: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_12 <= 1.1
 R3089: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x17_is_in_node_12 >= -1
 R3090: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x18_is_in_node_12 <= 1.1
 R3091: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_12 <= 1.1
 R3092: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x18_is_in_node_12 >= -1
 R3093: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x19_is_in_node_12 <= 1.1
 R3094: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_12 <= 1.1
 R3095: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x19_is_in_node_12 >= -1
 R3096: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x20_is_in_node_12 <= 1.1
 R3097: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_12 <= 1.1
 R3098: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x20_is_in_node_12 >= -1
 R3099: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x21_is_in_node_12 <= 1.1
 R3100: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_12 <= 1.1
 R3101: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.7 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x21_is_in_node_12 >= -1
 R3102: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.1 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x22_is_in_node_12 <= 1.1
 R3103: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_12 <= 1.1
 R3104: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.6 node_6_considers_var_1 + node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x22_is_in_node_12 >= -1
 R3105: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.6 node_3_considers_var_3 + 1.1 x23_is_in_node_12 <= 1.1
 R3106: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_12 <= 1.1
 R3107: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.5 node_6_considers_var_3 - x23_is_in_node_12 >= -1
 R3108: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x24_is_in_node_12 <= 1.1
 R3109: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_12 <= 1.1
 R3110: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.9 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x24_is_in_node_12 >= -1
 R3111: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x25_is_in_node_12 <= 1.1
 R3112: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_12 <= 1.1
 R3113: - split_value_node_6 + 5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x25_is_in_node_12 >= -1
 R3114: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x26_is_in_node_12 <= 1.1
 R3115: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_12 <= 1.1
 R3116: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x26_is_in_node_12 >= -1
 R3117: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x27_is_in_node_12 <= 1.1
 R3118: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_12 <= 1.1
 R3119: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x27_is_in_node_12 >= -1
 R3120: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x28_is_in_node_12 <= 1.1
 R3121: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_12 <= 1.1
 R3122: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x28_is_in_node_12 >= -1
 R3123: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x29_is_in_node_12 <= 1.1
 R3124: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_12 <= 1.1
 R3125: - split_value_node_6 + 4.7 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x29_is_in_node_12 >= -1
 R3126: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x30_is_in_node_12 <= 1.1
 R3127: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_12 <= 1.1
 R3128: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x30_is_in_node_12 >= -1
 R3129: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x31_is_in_node_12 <= 1.1
 R3130: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_12 <= 1.1
 R3131: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x31_is_in_node_12 >= -1
 R3132: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 4.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x32_is_in_node_12 <= 1.1
 R3133: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_12 <= 1.1
 R3134: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 4.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x32_is_in_node_12 >= -1
 R3135: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 4.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x33_is_in_node_12 <= 1.1
 R3136: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_12 <= 1.1
 R3137: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 4.2 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x33_is_in_node_12 >= -1
 R3138: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x34_is_in_node_12 <= 1.1
 R3139: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_12 <= 1.1
 R3140: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x34_is_in_node_12 >= -1
 R3141: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x35_is_in_node_12 <= 1.1
 R3142: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_12 <= 1.1
 R3143: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.2 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x35_is_in_node_12 >= -1
 R3144: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x36_is_in_node_12 <= 1.1
 R3145: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_12 <= 1.1
 R3146: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x36_is_in_node_12 >= -1
 R3147: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x37_is_in_node_12 <= 1.1
 R3148: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_12 <= 1.1
 R3149: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x37_is_in_node_12 >= -1
 R3150: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x38_is_in_node_12 <= 1.1
 R3151: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_12 <= 1.1
 R3152: - split_value_node_6 + 4.4 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x38_is_in_node_12 >= -1
 R3153: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x39_is_in_node_12 <= 1.1
 R3154: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_12 <= 1.1
 R3155: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x39_is_in_node_12 >= -1
 R3156: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x40_is_in_node_12 <= 1.1
 R3157: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_12 <= 1.1
 R3158: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x40_is_in_node_12 >= -1
 R3159: - split_value_node_3 + 4.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x41_is_in_node_12 <= 1.1
 R3160: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_12 <= 1.1
 R3161: - split_value_node_6 + 4.5 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x41_is_in_node_12 >= -1
 R3162: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x42_is_in_node_12 <= 1.1
 R3163: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_12 <= 1.1
 R3164: - split_value_node_6 + 4.4 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x42_is_in_node_12 >= -1
 R3165: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.7 node_3_considers_var_3 + 1.1 x43_is_in_node_12 <= 1.1
 R3166: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_12 <= 1.1
 R3167: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.6 node_6_considers_var_3 - x43_is_in_node_12 >= -1
 R3168: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x44_is_in_node_12 <= 1.1
 R3169: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_12 <= 1.1
 R3170: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.9 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x44_is_in_node_12 >= -1
 R3171: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x45_is_in_node_12 <= 1.1
 R3172: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_12 <= 1.1
 R3173: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x45_is_in_node_12 >= -1
 R3174: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x46_is_in_node_12 <= 1.1
 R3175: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_12 <= 1.1
 R3176: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x46_is_in_node_12 >= -1
 R3177: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x47_is_in_node_12 <= 1.1
 R3178: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_12 <= 1.1
 R3179: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x47_is_in_node_12 >= -1
 R3180: - split_value_node_3 + 5.4 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x48_is_in_node_12 <= 1.1
 R3181: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_12 <= 1.1
 R3182: - split_value_node_6 + 5.3 node_6_considers_var_0
   + 3.7 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x48_is_in_node_12 >= -1
 R3183: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x49_is_in_node_12 <= 1.1
 R3184: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_12 <= 1.1
 R3185: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x49_is_in_node_12 >= -1
 R3186: - split_value_node_3 + 7.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x50_is_in_node_12 <= 1.1
 R3187: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_12 <= 1.1
 R3188: - split_value_node_6 + 7 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x50_is_in_node_12 >= -1
 R3189: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x51_is_in_node_12 <= 1.1
 R3190: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_12 <= 1.1
 R3191: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x51_is_in_node_12 >= -1
 R3192: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x52_is_in_node_12 <= 1.1
 R3193: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_12 <= 1.1
 R3194: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x52_is_in_node_12 >= -1
 R3195: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x53_is_in_node_12 <= 1.1
 R3196: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_12 <= 1.1
 R3197: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x53_is_in_node_12 >= -1
 R3198: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x54_is_in_node_12 <= 1.1
 R3199: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_12 <= 1.1
 R3200: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.6 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x54_is_in_node_12 >= -1
 R3201: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x55_is_in_node_12 <= 1.1
 R3202: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_12 <= 1.1
 R3203: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x55_is_in_node_12 >= -1
 R3204: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x56_is_in_node_12 <= 1.1
 R3205: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_12 <= 1.1
 R3206: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x56_is_in_node_12 >= -1
 R3207: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x57_is_in_node_12 <= 1.1
 R3208: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_12 <= 1.1
 R3209: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 2.4 node_6_considers_var_1 + 3.3 node_6_considers_var_2
   + node_6_considers_var_3 - x57_is_in_node_12 >= -1
 R3210: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x58_is_in_node_12 <= 1.1
 R3211: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_12 <= 1.1
 R3212: - split_value_node_6 + 6.6 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.6 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x58_is_in_node_12 >= -1
 R3213: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x59_is_in_node_12 <= 1.1
 R3214: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_12 <= 1.1
 R3215: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 3.9 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x59_is_in_node_12 >= -1
 R3216: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.1 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x60_is_in_node_12 <= 1.1
 R3217: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_12 <= 1.1
 R3218: - split_value_node_6 + 5 node_6_considers_var_0
   + 2 node_6_considers_var_1 + 3.5 node_6_considers_var_2
   + node_6_considers_var_3 - x60_is_in_node_12 >= -1
 R3219: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x61_is_in_node_12 <= 1.1
 R3220: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_12 <= 1.1
 R3221: - split_value_node_6 + 5.9 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x61_is_in_node_12 >= -1
 R3222: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x62_is_in_node_12 <= 1.1
 R3223: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_12 <= 1.1
 R3224: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.2 node_6_considers_var_1 + 4 node_6_considers_var_2
   + node_6_considers_var_3 - x62_is_in_node_12 >= -1
 R3225: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x63_is_in_node_12 <= 1.1
 R3226: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_12 <= 1.1
 R3227: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x63_is_in_node_12 >= -1
 R3228: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 3.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x64_is_in_node_12 <= 1.1
 R3229: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_12 <= 1.1
 R3230: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 3.6 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x64_is_in_node_12 >= -1
 R3231: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x65_is_in_node_12 <= 1.1
 R3232: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_12 <= 1.1
 R3233: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x65_is_in_node_12 >= -1
 R3234: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x66_is_in_node_12 <= 1.1
 R3235: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_12 <= 1.1
 R3236: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x66_is_in_node_12 >= -1
 R3237: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x67_is_in_node_12 <= 1.1
 R3238: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_12 <= 1.1
 R3239: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 4.1 node_6_considers_var_2
   + node_6_considers_var_3 - x67_is_in_node_12 >= -1
 R3240: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x68_is_in_node_12 <= 1.1
 R3241: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_12 <= 1.1
 R3242: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 2.2 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x68_is_in_node_12 >= -1
 R3243: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x69_is_in_node_12 <= 1.1
 R3244: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_12 <= 1.1
 R3245: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 3.9 node_6_considers_var_2
   + 1.1 node_6_considers_var_3 - x69_is_in_node_12 >= -1
 R3246: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x70_is_in_node_12 <= 1.1
 R3247: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_12 <= 1.1
 R3248: - split_value_node_6 + 5.9 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x70_is_in_node_12 >= -1
 R3249: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x71_is_in_node_12 <= 1.1
 R3250: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_12 <= 1.1
 R3251: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x71_is_in_node_12 >= -1
 R3252: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x72_is_in_node_12 <= 1.1
 R3253: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_12 <= 1.1
 R3254: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x72_is_in_node_12 >= -1
 R3255: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x73_is_in_node_12 <= 1.1
 R3256: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_12 <= 1.1
 R3257: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x73_is_in_node_12 >= -1
 R3258: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x74_is_in_node_12 <= 1.1
 R3259: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_12 <= 1.1
 R3260: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.3 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x74_is_in_node_12 >= -1
 R3261: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x75_is_in_node_12 <= 1.1
 R3262: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_12 <= 1.1
 R3263: - split_value_node_6 + 6.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x75_is_in_node_12 >= -1
 R3264: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x76_is_in_node_12 <= 1.1
 R3265: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_12 <= 1.1
 R3266: - split_value_node_6 + 6.8 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x76_is_in_node_12 >= -1
 R3267: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x77_is_in_node_12 <= 1.1
 R3268: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_12 <= 1.1
 R3269: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 1.7 node_6_considers_var_3 - x77_is_in_node_12 >= -1
 R3270: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x78_is_in_node_12 <= 1.1
 R3271: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_12 <= 1.1
 R3272: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x78_is_in_node_12 >= -1
 R3273: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x79_is_in_node_12 <= 1.1
 R3274: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_12 <= 1.1
 R3275: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 3.5 node_6_considers_var_2
   + node_6_considers_var_3 - x79_is_in_node_12 >= -1
 R3276: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.9 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x80_is_in_node_12 <= 1.1
 R3277: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_12 <= 1.1
 R3278: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.4 node_6_considers_var_1 + 3.8 node_6_considers_var_2
   + 1.1 node_6_considers_var_3 - x80_is_in_node_12 >= -1
 R3279: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.8 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x81_is_in_node_12 <= 1.1
 R3280: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_12 <= 1.1
 R3281: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.4 node_6_considers_var_1 + 3.7 node_6_considers_var_2
   + node_6_considers_var_3 - x81_is_in_node_12 >= -1
 R3282: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x82_is_in_node_12 <= 1.1
 R3283: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_12 <= 1.1
 R3284: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 3.9 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x82_is_in_node_12 >= -1
 R3285: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x83_is_in_node_12 <= 1.1
 R3286: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_12 <= 1.1
 R3287: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x83_is_in_node_12 >= -1
 R3288: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x84_is_in_node_12 <= 1.1
 R3289: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_12 <= 1.1
 R3290: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x84_is_in_node_12 >= -1
 R3291: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x85_is_in_node_12 <= 1.1
 R3292: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_12 <= 1.1
 R3293: - split_value_node_6 + 6 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x85_is_in_node_12 >= -1
 R3294: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x86_is_in_node_12 <= 1.1
 R3295: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_12 <= 1.1
 R3296: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x86_is_in_node_12 >= -1
 R3297: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x87_is_in_node_12 <= 1.1
 R3298: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_12 <= 1.1
 R3299: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x87_is_in_node_12 >= -1
 R3300: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x88_is_in_node_12 <= 1.1
 R3301: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_12 <= 1.1
 R3302: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.1 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x88_is_in_node_12 >= -1
 R3303: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x89_is_in_node_12 <= 1.1
 R3304: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_12 <= 1.1
 R3305: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x89_is_in_node_12 >= -1
 R3306: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x90_is_in_node_12 <= 1.1
 R3307: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_12 <= 1.1
 R3308: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x90_is_in_node_12 >= -1
 R3309: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x91_is_in_node_12 <= 1.1
 R3310: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_12 <= 1.1
 R3311: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.6 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x91_is_in_node_12 >= -1
 R3312: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x92_is_in_node_12 <= 1.1
 R3313: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_12 <= 1.1
 R3314: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x92_is_in_node_12 >= -1
 R3315: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x93_is_in_node_12 <= 1.1
 R3316: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_12 <= 1.1
 R3317: - split_value_node_6 + 5 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 3.3 node_6_considers_var_2
   + node_6_considers_var_3 - x93_is_in_node_12 >= -1
 R3318: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x94_is_in_node_12 <= 1.1
 R3319: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_12 <= 1.1
 R3320: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x94_is_in_node_12 >= -1
 R3321: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x95_is_in_node_12 <= 1.1
 R3322: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_12 <= 1.1
 R3323: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x95_is_in_node_12 >= -1
 R3324: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x96_is_in_node_12 <= 1.1
 R3325: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_12 <= 1.1
 R3326: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x96_is_in_node_12 >= -1
 R3327: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x97_is_in_node_12 <= 1.1
 R3328: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_12 <= 1.1
 R3329: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.3 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x97_is_in_node_12 >= -1
 R3330: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 3.1 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x98_is_in_node_12 <= 1.1
 R3331: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_12 <= 1.1
 R3332: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 3 node_6_considers_var_2
   + 1.1 node_6_considers_var_3 - x98_is_in_node_12 >= -1
 R3333: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x99_is_in_node_12 <= 1.1
 R3334: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_12 <= 1.1
 R3335: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.1 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x99_is_in_node_12 >= -1
 R3336: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x100_is_in_node_12 <= 1.1
 R3337: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_12 <= 1.1
 R3338: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 6 node_6_considers_var_2
   + 2.5 node_6_considers_var_3 - x100_is_in_node_12 >= -1
 R3339: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x101_is_in_node_12 <= 1.1
 R3340: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_12 <= 1.1
 R3341: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x101_is_in_node_12 >= -1
 R3342: - split_value_node_3 + 7.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x102_is_in_node_12 <= 1.1
 R3343: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_12 <= 1.1
 R3344: - split_value_node_6 + 7.1 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.9 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x102_is_in_node_12 >= -1
 R3345: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x103_is_in_node_12 <= 1.1
 R3346: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_12 <= 1.1
 R3347: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x103_is_in_node_12 >= -1
 R3348: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x104_is_in_node_12 <= 1.1
 R3349: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_12 <= 1.1
 R3350: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.8 node_6_considers_var_2
   + 2.2 node_6_considers_var_3 - x104_is_in_node_12 >= -1
 R3351: - split_value_node_3 + 7.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x105_is_in_node_12 <= 1.1
 R3352: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_12 <= 1.1
 R3353: - split_value_node_6 + 7.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 6.6 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x105_is_in_node_12 >= -1
 R3354: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x106_is_in_node_12 <= 1.1
 R3355: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_12 <= 1.1
 R3356: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.7 node_6_considers_var_3 - x106_is_in_node_12 >= -1
 R3357: - split_value_node_3 + 7.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 6.4 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x107_is_in_node_12 <= 1.1
 R3358: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_12 <= 1.1
 R3359: - split_value_node_6 + 7.3 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 6.3 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x107_is_in_node_12 >= -1
 R3360: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x108_is_in_node_12 <= 1.1
 R3361: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_12 <= 1.1
 R3362: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 5.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x108_is_in_node_12 >= -1
 R3363: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x109_is_in_node_12 <= 1.1
 R3364: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_12 <= 1.1
 R3365: - split_value_node_6 + 7.2 node_6_considers_var_0
   + 3.6 node_6_considers_var_1 + 6.1 node_6_considers_var_2
   + 2.5 node_6_considers_var_3 - x109_is_in_node_12 >= -1
 R3366: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x110_is_in_node_12 <= 1.1
 R3367: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_12 <= 1.1
 R3368: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x110_is_in_node_12 >= -1
 R3369: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x111_is_in_node_12 <= 1.1
 R3370: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_12 <= 1.1
 R3371: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.3 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x111_is_in_node_12 >= -1
 R3372: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x112_is_in_node_12 <= 1.1
 R3373: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_12 <= 1.1
 R3374: - split_value_node_6 + 6.8 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.5 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x112_is_in_node_12 >= -1
 R3375: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x113_is_in_node_12 <= 1.1
 R3376: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_12 <= 1.1
 R3377: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x113_is_in_node_12 >= -1
 R3378: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x114_is_in_node_12 <= 1.1
 R3379: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_12 <= 1.1
 R3380: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 2.4 node_6_considers_var_3 - x114_is_in_node_12 >= -1
 R3381: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x115_is_in_node_12 <= 1.1
 R3382: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_12 <= 1.1
 R3383: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.3 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x115_is_in_node_12 >= -1
 R3384: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x116_is_in_node_12 <= 1.1
 R3385: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_12 <= 1.1
 R3386: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.5 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x116_is_in_node_12 >= -1
 R3387: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x117_is_in_node_12 <= 1.1
 R3388: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_12 <= 1.1
 R3389: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 6.7 node_6_considers_var_2
   + 2.2 node_6_considers_var_3 - x117_is_in_node_12 >= -1
 R3390: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 7 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x118_is_in_node_12 <= 1.1
 R3391: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_12 <= 1.1
 R3392: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 6.9 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x118_is_in_node_12 >= -1
 R3393: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x119_is_in_node_12 <= 1.1
 R3394: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_12 <= 1.1
 R3395: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.2 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x119_is_in_node_12 >= -1
 R3396: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x120_is_in_node_12 <= 1.1
 R3397: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_12 <= 1.1
 R3398: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.7 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x120_is_in_node_12 >= -1
 R3399: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x121_is_in_node_12 <= 1.1
 R3400: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_12 <= 1.1
 R3401: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x121_is_in_node_12 >= -1
 R3402: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x122_is_in_node_12 <= 1.1
 R3403: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_12 <= 1.1
 R3404: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 6.7 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x122_is_in_node_12 >= -1
 R3405: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x123_is_in_node_12 <= 1.1
 R3406: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_12 <= 1.1
 R3407: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x123_is_in_node_12 >= -1
 R3408: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x124_is_in_node_12 <= 1.1
 R3409: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_12 <= 1.1
 R3410: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 5.7 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x124_is_in_node_12 >= -1
 R3411: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x125_is_in_node_12 <= 1.1
 R3412: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_12 <= 1.1
 R3413: - split_value_node_6 + 7.2 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 6 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x125_is_in_node_12 >= -1
 R3414: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x126_is_in_node_12 <= 1.1
 R3415: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_12 <= 1.1
 R3416: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x126_is_in_node_12 >= -1
 R3417: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x127_is_in_node_12 <= 1.1
 R3418: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_12 <= 1.1
 R3419: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x127_is_in_node_12 >= -1
 R3420: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x128_is_in_node_12 <= 1.1
 R3421: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_12 <= 1.1
 R3422: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x128_is_in_node_12 >= -1
 R3423: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x129_is_in_node_12 <= 1.1
 R3424: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_12 <= 1.1
 R3425: - split_value_node_6 + 7.2 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.8 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x129_is_in_node_12 >= -1
 R3426: - split_value_node_3 + 7.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x130_is_in_node_12 <= 1.1
 R3427: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_12 <= 1.1
 R3428: - split_value_node_6 + 7.4 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 6.1 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x130_is_in_node_12 >= -1
 R3429: - split_value_node_3 + 8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x131_is_in_node_12 <= 1.1
 R3430: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_12 <= 1.1
 R3431: - split_value_node_6 + 7.9 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 6.4 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x131_is_in_node_12 >= -1
 R3432: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x132_is_in_node_12 <= 1.1
 R3433: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_12 <= 1.1
 R3434: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.2 node_6_considers_var_3 - x132_is_in_node_12 >= -1
 R3435: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x133_is_in_node_12 <= 1.1
 R3436: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_12 <= 1.1
 R3437: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x133_is_in_node_12 >= -1
 R3438: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x134_is_in_node_12 <= 1.1
 R3439: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_12 <= 1.1
 R3440: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x134_is_in_node_12 >= -1
 R3441: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x135_is_in_node_12 <= 1.1
 R3442: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_12 <= 1.1
 R3443: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 6.1 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x135_is_in_node_12 >= -1
 R3444: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x136_is_in_node_12 <= 1.1
 R3445: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_12 <= 1.1
 R3446: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.4 node_6_considers_var_3 - x136_is_in_node_12 >= -1
 R3447: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x137_is_in_node_12 <= 1.1
 R3448: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_12 <= 1.1
 R3449: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.5 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x137_is_in_node_12 >= -1
 R3450: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x138_is_in_node_12 <= 1.1
 R3451: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_12 <= 1.1
 R3452: - split_value_node_6 + 6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x138_is_in_node_12 >= -1
 R3453: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x139_is_in_node_12 <= 1.1
 R3454: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_12 <= 1.1
 R3455: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.4 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x139_is_in_node_12 >= -1
 R3456: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x140_is_in_node_12 <= 1.1
 R3457: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_12 <= 1.1
 R3458: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.4 node_6_considers_var_3 - x140_is_in_node_12 >= -1
 R3459: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x141_is_in_node_12 <= 1.1
 R3460: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_12 <= 1.1
 R3461: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x141_is_in_node_12 >= -1
 R3462: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x142_is_in_node_12 <= 1.1
 R3463: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_12 <= 1.1
 R3464: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x142_is_in_node_12 >= -1
 R3465: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x143_is_in_node_12 <= 1.1
 R3466: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_12 <= 1.1
 R3467: - split_value_node_6 + 6.8 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.9 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x143_is_in_node_12 >= -1
 R3468: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x144_is_in_node_12 <= 1.1
 R3469: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_12 <= 1.1
 R3470: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 5.7 node_6_considers_var_2
   + 2.5 node_6_considers_var_3 - x144_is_in_node_12 >= -1
 R3471: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x145_is_in_node_12 <= 1.1
 R3472: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_12 <= 1.1
 R3473: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.2 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x145_is_in_node_12 >= -1
 R3474: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x146_is_in_node_12 <= 1.1
 R3475: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_12 <= 1.1
 R3476: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x146_is_in_node_12 >= -1
 R3477: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x147_is_in_node_12 <= 1.1
 R3478: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_12 <= 1.1
 R3479: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.2 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x147_is_in_node_12 >= -1
 R3480: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x148_is_in_node_12 <= 1.1
 R3481: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_12 <= 1.1
 R3482: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 5.4 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x148_is_in_node_12 >= -1
 R3483: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x149_is_in_node_12 <= 1.1
 R3484: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_12 <= 1.1
 R3485: - split_value_node_6 + 5.9 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x149_is_in_node_12 >= -1
 R3486: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x0_is_in_node_13 <= 1.1
 R3487: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_13 <= 1.1
 R3488: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x0_is_in_node_13 >= -1
 R3489: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x1_is_in_node_13 <= 1.1
 R3490: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_13 <= 1.1
 R3491: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x1_is_in_node_13 >= -1
 R3492: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x2_is_in_node_13 <= 1.1
 R3493: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_13 <= 1.1
 R3494: - split_value_node_6 + 4.7 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x2_is_in_node_13 >= -1
 R3495: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x3_is_in_node_13 <= 1.1
 R3496: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_13 <= 1.1
 R3497: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x3_is_in_node_13 >= -1
 R3498: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x4_is_in_node_13 <= 1.1
 R3499: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_13 <= 1.1
 R3500: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.6 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x4_is_in_node_13 >= -1
 R3501: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x5_is_in_node_13 <= 1.1
 R3502: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_13 <= 1.1
 R3503: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.9 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x5_is_in_node_13 >= -1
 R3504: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x6_is_in_node_13 <= 1.1
 R3505: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_13 <= 1.1
 R3506: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x6_is_in_node_13 >= -1
 R3507: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x7_is_in_node_13 <= 1.1
 R3508: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_13 <= 1.1
 R3509: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x7_is_in_node_13 >= -1
 R3510: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x8_is_in_node_13 <= 1.1
 R3511: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_13 <= 1.1
 R3512: - split_value_node_6 + 4.4 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x8_is_in_node_13 >= -1
 R3513: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x9_is_in_node_13 <= 1.1
 R3514: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_13 <= 1.1
 R3515: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x9_is_in_node_13 >= -1
 R3516: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x10_is_in_node_13 <= 1.1
 R3517: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_13 <= 1.1
 R3518: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.7 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x10_is_in_node_13 >= -1
 R3519: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x11_is_in_node_13 <= 1.1
 R3520: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_13 <= 1.1
 R3521: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x11_is_in_node_13 >= -1
 R3522: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x12_is_in_node_13 <= 1.1
 R3523: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_13 <= 1.1
 R3524: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x12_is_in_node_13 >= -1
 R3525: - split_value_node_3 + 4.4 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.2 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x13_is_in_node_13 <= 1.1
 R3526: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_13 <= 1.1
 R3527: - split_value_node_6 + 4.3 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.1 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x13_is_in_node_13 >= -1
 R3528: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 4.1 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x14_is_in_node_13 <= 1.1
 R3529: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_13 <= 1.1
 R3530: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 4 node_6_considers_var_1 + 1.2 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x14_is_in_node_13 >= -1
 R3531: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 4.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x15_is_in_node_13 <= 1.1
 R3532: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_13 <= 1.1
 R3533: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 4.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x15_is_in_node_13 >= -1
 R3534: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x16_is_in_node_13 <= 1.1
 R3535: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_13 <= 1.1
 R3536: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.9 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x16_is_in_node_13 >= -1
 R3537: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x17_is_in_node_13 <= 1.1
 R3538: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_13 <= 1.1
 R3539: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x17_is_in_node_13 >= -1
 R3540: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x18_is_in_node_13 <= 1.1
 R3541: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_13 <= 1.1
 R3542: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x18_is_in_node_13 >= -1
 R3543: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x19_is_in_node_13 <= 1.1
 R3544: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_13 <= 1.1
 R3545: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x19_is_in_node_13 >= -1
 R3546: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x20_is_in_node_13 <= 1.1
 R3547: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_13 <= 1.1
 R3548: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x20_is_in_node_13 >= -1
 R3549: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x21_is_in_node_13 <= 1.1
 R3550: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_13 <= 1.1
 R3551: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.7 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x21_is_in_node_13 >= -1
 R3552: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.1 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x22_is_in_node_13 <= 1.1
 R3553: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_13 <= 1.1
 R3554: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.6 node_6_considers_var_1 + node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x22_is_in_node_13 >= -1
 R3555: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.6 node_3_considers_var_3 + 1.1 x23_is_in_node_13 <= 1.1
 R3556: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_13 <= 1.1
 R3557: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 1.7 node_6_considers_var_2
   + 0.5 node_6_considers_var_3 - x23_is_in_node_13 >= -1
 R3558: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x24_is_in_node_13 <= 1.1
 R3559: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_13 <= 1.1
 R3560: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.9 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x24_is_in_node_13 >= -1
 R3561: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x25_is_in_node_13 <= 1.1
 R3562: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_13 <= 1.1
 R3563: - split_value_node_6 + 5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x25_is_in_node_13 >= -1
 R3564: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x26_is_in_node_13 <= 1.1
 R3565: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_13 <= 1.1
 R3566: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x26_is_in_node_13 >= -1
 R3567: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x27_is_in_node_13 <= 1.1
 R3568: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_13 <= 1.1
 R3569: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x27_is_in_node_13 >= -1
 R3570: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x28_is_in_node_13 <= 1.1
 R3571: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_13 <= 1.1
 R3572: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x28_is_in_node_13 >= -1
 R3573: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x29_is_in_node_13 <= 1.1
 R3574: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_13 <= 1.1
 R3575: - split_value_node_6 + 4.7 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x29_is_in_node_13 >= -1
 R3576: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x30_is_in_node_13 <= 1.1
 R3577: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_13 <= 1.1
 R3578: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x30_is_in_node_13 >= -1
 R3579: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x31_is_in_node_13 <= 1.1
 R3580: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_13 <= 1.1
 R3581: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x31_is_in_node_13 >= -1
 R3582: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 4.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x32_is_in_node_13 <= 1.1
 R3583: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_13 <= 1.1
 R3584: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 4.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x32_is_in_node_13 >= -1
 R3585: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 4.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x33_is_in_node_13 <= 1.1
 R3586: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_13 <= 1.1
 R3587: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 4.2 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x33_is_in_node_13 >= -1
 R3588: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x34_is_in_node_13 <= 1.1
 R3589: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_13 <= 1.1
 R3590: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x34_is_in_node_13 >= -1
 R3591: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x35_is_in_node_13 <= 1.1
 R3592: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_13 <= 1.1
 R3593: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.2 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x35_is_in_node_13 >= -1
 R3594: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x36_is_in_node_13 <= 1.1
 R3595: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_13 <= 1.1
 R3596: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x36_is_in_node_13 >= -1
 R3597: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x37_is_in_node_13 <= 1.1
 R3598: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_13 <= 1.1
 R3599: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.1 node_6_considers_var_3 - x37_is_in_node_13 >= -1
 R3600: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x38_is_in_node_13 <= 1.1
 R3601: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_13 <= 1.1
 R3602: - split_value_node_6 + 4.4 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x38_is_in_node_13 >= -1
 R3603: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x39_is_in_node_13 <= 1.1
 R3604: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_13 <= 1.1
 R3605: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x39_is_in_node_13 >= -1
 R3606: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x40_is_in_node_13 <= 1.1
 R3607: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_13 <= 1.1
 R3608: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x40_is_in_node_13 >= -1
 R3609: - split_value_node_3 + 4.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x41_is_in_node_13 <= 1.1
 R3610: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_13 <= 1.1
 R3611: - split_value_node_6 + 4.5 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x41_is_in_node_13 >= -1
 R3612: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x42_is_in_node_13 <= 1.1
 R3613: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_13 <= 1.1
 R3614: - split_value_node_6 + 4.4 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.3 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x42_is_in_node_13 >= -1
 R3615: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.7 node_3_considers_var_3 + 1.1 x43_is_in_node_13 <= 1.1
 R3616: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_13 <= 1.1
 R3617: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.5 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.6 node_6_considers_var_3 - x43_is_in_node_13 >= -1
 R3618: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x44_is_in_node_13 <= 1.1
 R3619: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_13 <= 1.1
 R3620: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.9 node_6_considers_var_2
   + 0.4 node_6_considers_var_3 - x44_is_in_node_13 >= -1
 R3621: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x45_is_in_node_13 <= 1.1
 R3622: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_13 <= 1.1
 R3623: - split_value_node_6 + 4.8 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.3 node_6_considers_var_3 - x45_is_in_node_13 >= -1
 R3624: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x46_is_in_node_13 <= 1.1
 R3625: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_13 <= 1.1
 R3626: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 1.6 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x46_is_in_node_13 >= -1
 R3627: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x47_is_in_node_13 <= 1.1
 R3628: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_13 <= 1.1
 R3629: - split_value_node_6 + 4.6 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x47_is_in_node_13 >= -1
 R3630: - split_value_node_3 + 5.4 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x48_is_in_node_13 <= 1.1
 R3631: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_13 <= 1.1
 R3632: - split_value_node_6 + 5.3 node_6_considers_var_0
   + 3.7 node_6_considers_var_1 + 1.5 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x48_is_in_node_13 >= -1
 R3633: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x49_is_in_node_13 <= 1.1
 R3634: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_13 <= 1.1
 R3635: - split_value_node_6 + 5 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 1.4 node_6_considers_var_2
   + 0.2 node_6_considers_var_3 - x49_is_in_node_13 >= -1
 R3636: - split_value_node_3 + 7.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x50_is_in_node_13 <= 1.1
 R3637: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_13 <= 1.1
 R3638: - split_value_node_6 + 7 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x50_is_in_node_13 >= -1
 R3639: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x51_is_in_node_13 <= 1.1
 R3640: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_13 <= 1.1
 R3641: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x51_is_in_node_13 >= -1
 R3642: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x52_is_in_node_13 <= 1.1
 R3643: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_13 <= 1.1
 R3644: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x52_is_in_node_13 >= -1
 R3645: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x53_is_in_node_13 <= 1.1
 R3646: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_13 <= 1.1
 R3647: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x53_is_in_node_13 >= -1
 R3648: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x54_is_in_node_13 <= 1.1
 R3649: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_13 <= 1.1
 R3650: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.6 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x54_is_in_node_13 >= -1
 R3651: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x55_is_in_node_13 <= 1.1
 R3652: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_13 <= 1.1
 R3653: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x55_is_in_node_13 >= -1
 R3654: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x56_is_in_node_13 <= 1.1
 R3655: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_13 <= 1.1
 R3656: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x56_is_in_node_13 >= -1
 R3657: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x57_is_in_node_13 <= 1.1
 R3658: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_13 <= 1.1
 R3659: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 2.4 node_6_considers_var_1 + 3.3 node_6_considers_var_2
   + node_6_considers_var_3 - x57_is_in_node_13 >= -1
 R3660: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x58_is_in_node_13 <= 1.1
 R3661: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_13 <= 1.1
 R3662: - split_value_node_6 + 6.6 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.6 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x58_is_in_node_13 >= -1
 R3663: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x59_is_in_node_13 <= 1.1
 R3664: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_13 <= 1.1
 R3665: - split_value_node_6 + 5.2 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 3.9 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x59_is_in_node_13 >= -1
 R3666: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.1 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x60_is_in_node_13 <= 1.1
 R3667: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_13 <= 1.1
 R3668: - split_value_node_6 + 5 node_6_considers_var_0
   + 2 node_6_considers_var_1 + 3.5 node_6_considers_var_2
   + node_6_considers_var_3 - x60_is_in_node_13 >= -1
 R3669: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x61_is_in_node_13 <= 1.1
 R3670: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_13 <= 1.1
 R3671: - split_value_node_6 + 5.9 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x61_is_in_node_13 >= -1
 R3672: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x62_is_in_node_13 <= 1.1
 R3673: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_13 <= 1.1
 R3674: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.2 node_6_considers_var_1 + 4 node_6_considers_var_2
   + node_6_considers_var_3 - x62_is_in_node_13 >= -1
 R3675: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x63_is_in_node_13 <= 1.1
 R3676: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_13 <= 1.1
 R3677: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x63_is_in_node_13 >= -1
 R3678: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 3.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x64_is_in_node_13 <= 1.1
 R3679: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_13 <= 1.1
 R3680: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 3.6 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x64_is_in_node_13 >= -1
 R3681: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x65_is_in_node_13 <= 1.1
 R3682: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_13 <= 1.1
 R3683: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x65_is_in_node_13 >= -1
 R3684: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x66_is_in_node_13 <= 1.1
 R3685: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_13 <= 1.1
 R3686: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x66_is_in_node_13 >= -1
 R3687: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x67_is_in_node_13 <= 1.1
 R3688: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_13 <= 1.1
 R3689: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 4.1 node_6_considers_var_2
   + node_6_considers_var_3 - x67_is_in_node_13 >= -1
 R3690: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x68_is_in_node_13 <= 1.1
 R3691: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_13 <= 1.1
 R3692: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 2.2 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x68_is_in_node_13 >= -1
 R3693: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x69_is_in_node_13 <= 1.1
 R3694: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_13 <= 1.1
 R3695: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 3.9 node_6_considers_var_2
   + 1.1 node_6_considers_var_3 - x69_is_in_node_13 >= -1
 R3696: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x70_is_in_node_13 <= 1.1
 R3697: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_13 <= 1.1
 R3698: - split_value_node_6 + 5.9 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x70_is_in_node_13 >= -1
 R3699: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x71_is_in_node_13 <= 1.1
 R3700: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_13 <= 1.1
 R3701: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x71_is_in_node_13 >= -1
 R3702: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x72_is_in_node_13 <= 1.1
 R3703: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_13 <= 1.1
 R3704: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x72_is_in_node_13 >= -1
 R3705: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x73_is_in_node_13 <= 1.1
 R3706: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_13 <= 1.1
 R3707: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x73_is_in_node_13 >= -1
 R3708: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x74_is_in_node_13 <= 1.1
 R3709: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_13 <= 1.1
 R3710: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.3 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x74_is_in_node_13 >= -1
 R3711: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x75_is_in_node_13 <= 1.1
 R3712: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_13 <= 1.1
 R3713: - split_value_node_6 + 6.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x75_is_in_node_13 >= -1
 R3714: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x76_is_in_node_13 <= 1.1
 R3715: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_13 <= 1.1
 R3716: - split_value_node_6 + 6.8 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x76_is_in_node_13 >= -1
 R3717: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x77_is_in_node_13 <= 1.1
 R3718: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_13 <= 1.1
 R3719: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 1.7 node_6_considers_var_3 - x77_is_in_node_13 >= -1
 R3720: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x78_is_in_node_13 <= 1.1
 R3721: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_13 <= 1.1
 R3722: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x78_is_in_node_13 >= -1
 R3723: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x79_is_in_node_13 <= 1.1
 R3724: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_13 <= 1.1
 R3725: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 3.5 node_6_considers_var_2
   + node_6_considers_var_3 - x79_is_in_node_13 >= -1
 R3726: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.9 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x80_is_in_node_13 <= 1.1
 R3727: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_13 <= 1.1
 R3728: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.4 node_6_considers_var_1 + 3.8 node_6_considers_var_2
   + 1.1 node_6_considers_var_3 - x80_is_in_node_13 >= -1
 R3729: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.8 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x81_is_in_node_13 <= 1.1
 R3730: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_13 <= 1.1
 R3731: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.4 node_6_considers_var_1 + 3.7 node_6_considers_var_2
   + node_6_considers_var_3 - x81_is_in_node_13 >= -1
 R3732: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x82_is_in_node_13 <= 1.1
 R3733: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_13 <= 1.1
 R3734: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 3.9 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x82_is_in_node_13 >= -1
 R3735: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x83_is_in_node_13 <= 1.1
 R3736: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_13 <= 1.1
 R3737: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x83_is_in_node_13 >= -1
 R3738: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x84_is_in_node_13 <= 1.1
 R3739: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_13 <= 1.1
 R3740: - split_value_node_6 + 5.4 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x84_is_in_node_13 >= -1
 R3741: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x85_is_in_node_13 <= 1.1
 R3742: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_13 <= 1.1
 R3743: - split_value_node_6 + 6 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x85_is_in_node_13 >= -1
 R3744: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x86_is_in_node_13 <= 1.1
 R3745: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_13 <= 1.1
 R3746: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 4.7 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x86_is_in_node_13 >= -1
 R3747: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x87_is_in_node_13 <= 1.1
 R3748: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_13 <= 1.1
 R3749: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x87_is_in_node_13 >= -1
 R3750: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x88_is_in_node_13 <= 1.1
 R3751: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_13 <= 1.1
 R3752: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.1 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x88_is_in_node_13 >= -1
 R3753: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x89_is_in_node_13 <= 1.1
 R3754: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_13 <= 1.1
 R3755: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x89_is_in_node_13 >= -1
 R3756: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x90_is_in_node_13 <= 1.1
 R3757: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_13 <= 1.1
 R3758: - split_value_node_6 + 5.5 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 4.4 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x90_is_in_node_13 >= -1
 R3759: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x91_is_in_node_13 <= 1.1
 R3760: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_13 <= 1.1
 R3761: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.6 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x91_is_in_node_13 >= -1
 R3762: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x92_is_in_node_13 <= 1.1
 R3763: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_13 <= 1.1
 R3764: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 4 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x92_is_in_node_13 >= -1
 R3765: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x93_is_in_node_13 <= 1.1
 R3766: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_13 <= 1.1
 R3767: - split_value_node_6 + 5 node_6_considers_var_0
   + 2.3 node_6_considers_var_1 + 3.3 node_6_considers_var_2
   + node_6_considers_var_3 - x93_is_in_node_13 >= -1
 R3768: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x94_is_in_node_13 <= 1.1
 R3769: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_13 <= 1.1
 R3770: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x94_is_in_node_13 >= -1
 R3771: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x95_is_in_node_13 <= 1.1
 R3772: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_13 <= 1.1
 R3773: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.2 node_6_considers_var_3 - x95_is_in_node_13 >= -1
 R3774: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x96_is_in_node_13 <= 1.1
 R3775: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_13 <= 1.1
 R3776: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.2 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x96_is_in_node_13 >= -1
 R3777: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x97_is_in_node_13 <= 1.1
 R3778: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_13 <= 1.1
 R3779: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 4.3 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x97_is_in_node_13 >= -1
 R3780: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 3.1 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x98_is_in_node_13 <= 1.1
 R3781: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_13 <= 1.1
 R3782: - split_value_node_6 + 5.1 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 3 node_6_considers_var_2
   + 1.1 node_6_considers_var_3 - x98_is_in_node_13 >= -1
 R3783: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x99_is_in_node_13 <= 1.1
 R3784: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_13 <= 1.1
 R3785: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.1 node_6_considers_var_2
   + 1.3 node_6_considers_var_3 - x99_is_in_node_13 >= -1
 R3786: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x100_is_in_node_13 <= 1.1
 R3787: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_13 <= 1.1
 R3788: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 6 node_6_considers_var_2
   + 2.5 node_6_considers_var_3 - x100_is_in_node_13 >= -1
 R3789: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x101_is_in_node_13 <= 1.1
 R3790: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_13 <= 1.1
 R3791: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x101_is_in_node_13 >= -1
 R3792: - split_value_node_3 + 7.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x102_is_in_node_13 <= 1.1
 R3793: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_13 <= 1.1
 R3794: - split_value_node_6 + 7.1 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.9 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x102_is_in_node_13 >= -1
 R3795: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x103_is_in_node_13 <= 1.1
 R3796: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_13 <= 1.1
 R3797: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x103_is_in_node_13 >= -1
 R3798: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x104_is_in_node_13 <= 1.1
 R3799: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_13 <= 1.1
 R3800: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.8 node_6_considers_var_2
   + 2.2 node_6_considers_var_3 - x104_is_in_node_13 >= -1
 R3801: - split_value_node_3 + 7.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x105_is_in_node_13 <= 1.1
 R3802: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_13 <= 1.1
 R3803: - split_value_node_6 + 7.6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 6.6 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x105_is_in_node_13 >= -1
 R3804: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x106_is_in_node_13 <= 1.1
 R3805: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_13 <= 1.1
 R3806: - split_value_node_6 + 4.9 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 4.5 node_6_considers_var_2
   + 1.7 node_6_considers_var_3 - x106_is_in_node_13 >= -1
 R3807: - split_value_node_3 + 7.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 6.4 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x107_is_in_node_13 <= 1.1
 R3808: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_13 <= 1.1
 R3809: - split_value_node_6 + 7.3 node_6_considers_var_0
   + 2.9 node_6_considers_var_1 + 6.3 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x107_is_in_node_13 >= -1
 R3810: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x108_is_in_node_13 <= 1.1
 R3811: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_13 <= 1.1
 R3812: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 5.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x108_is_in_node_13 >= -1
 R3813: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x109_is_in_node_13 <= 1.1
 R3814: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_13 <= 1.1
 R3815: - split_value_node_6 + 7.2 node_6_considers_var_0
   + 3.6 node_6_considers_var_1 + 6.1 node_6_considers_var_2
   + 2.5 node_6_considers_var_3 - x109_is_in_node_13 >= -1
 R3816: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x110_is_in_node_13 <= 1.1
 R3817: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_13 <= 1.1
 R3818: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x110_is_in_node_13 >= -1
 R3819: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x111_is_in_node_13 <= 1.1
 R3820: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_13 <= 1.1
 R3821: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.3 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x111_is_in_node_13 >= -1
 R3822: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x112_is_in_node_13 <= 1.1
 R3823: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_13 <= 1.1
 R3824: - split_value_node_6 + 6.8 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.5 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x112_is_in_node_13 >= -1
 R3825: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x113_is_in_node_13 <= 1.1
 R3826: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_13 <= 1.1
 R3827: - split_value_node_6 + 5.7 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x113_is_in_node_13 >= -1
 R3828: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x114_is_in_node_13 <= 1.1
 R3829: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_13 <= 1.1
 R3830: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 2.4 node_6_considers_var_3 - x114_is_in_node_13 >= -1
 R3831: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x115_is_in_node_13 <= 1.1
 R3832: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_13 <= 1.1
 R3833: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.3 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x115_is_in_node_13 >= -1
 R3834: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x116_is_in_node_13 <= 1.1
 R3835: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_13 <= 1.1
 R3836: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.5 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x116_is_in_node_13 >= -1
 R3837: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x117_is_in_node_13 <= 1.1
 R3838: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_13 <= 1.1
 R3839: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 6.7 node_6_considers_var_2
   + 2.2 node_6_considers_var_3 - x117_is_in_node_13 >= -1
 R3840: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 7 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x118_is_in_node_13 <= 1.1
 R3841: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_13 <= 1.1
 R3842: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 6.9 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x118_is_in_node_13 >= -1
 R3843: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x119_is_in_node_13 <= 1.1
 R3844: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_13 <= 1.1
 R3845: - split_value_node_6 + 6 node_6_considers_var_0
   + 2.2 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x119_is_in_node_13 >= -1
 R3846: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x120_is_in_node_13 <= 1.1
 R3847: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_13 <= 1.1
 R3848: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.7 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x120_is_in_node_13 >= -1
 R3849: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x121_is_in_node_13 <= 1.1
 R3850: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_13 <= 1.1
 R3851: - split_value_node_6 + 5.6 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x121_is_in_node_13 >= -1
 R3852: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x122_is_in_node_13 <= 1.1
 R3853: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_13 <= 1.1
 R3854: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 6.7 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x122_is_in_node_13 >= -1
 R3855: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x123_is_in_node_13 <= 1.1
 R3856: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_13 <= 1.1
 R3857: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x123_is_in_node_13 >= -1
 R3858: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x124_is_in_node_13 <= 1.1
 R3859: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_13 <= 1.1
 R3860: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 5.7 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x124_is_in_node_13 >= -1
 R3861: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x125_is_in_node_13 <= 1.1
 R3862: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_13 <= 1.1
 R3863: - split_value_node_6 + 7.2 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 6 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x125_is_in_node_13 >= -1
 R3864: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x126_is_in_node_13 <= 1.1
 R3865: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_13 <= 1.1
 R3866: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x126_is_in_node_13 >= -1
 R3867: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x127_is_in_node_13 <= 1.1
 R3868: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_13 <= 1.1
 R3869: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.9 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x127_is_in_node_13 >= -1
 R3870: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x128_is_in_node_13 <= 1.1
 R3871: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_13 <= 1.1
 R3872: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x128_is_in_node_13 >= -1
 R3873: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x129_is_in_node_13 <= 1.1
 R3874: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_13 <= 1.1
 R3875: - split_value_node_6 + 7.2 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.8 node_6_considers_var_2
   + 1.6 node_6_considers_var_3 - x129_is_in_node_13 >= -1
 R3876: - split_value_node_3 + 7.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x130_is_in_node_13 <= 1.1
 R3877: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_13 <= 1.1
 R3878: - split_value_node_6 + 7.4 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 6.1 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x130_is_in_node_13 >= -1
 R3879: - split_value_node_3 + 8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x131_is_in_node_13 <= 1.1
 R3880: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_13 <= 1.1
 R3881: - split_value_node_6 + 7.9 node_6_considers_var_0
   + 3.8 node_6_considers_var_1 + 6.4 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x131_is_in_node_13 >= -1
 R3882: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x132_is_in_node_13 <= 1.1
 R3883: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_13 <= 1.1
 R3884: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.2 node_6_considers_var_3 - x132_is_in_node_13 >= -1
 R3885: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x133_is_in_node_13 <= 1.1
 R3886: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_13 <= 1.1
 R3887: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.8 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.5 node_6_considers_var_3 - x133_is_in_node_13 >= -1
 R3888: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x134_is_in_node_13 <= 1.1
 R3889: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_13 <= 1.1
 R3890: - split_value_node_6 + 6.1 node_6_considers_var_0
   + 2.6 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 1.4 node_6_considers_var_3 - x134_is_in_node_13 >= -1
 R3891: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x135_is_in_node_13 <= 1.1
 R3892: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_13 <= 1.1
 R3893: - split_value_node_6 + 7.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 6.1 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x135_is_in_node_13 >= -1
 R3894: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x136_is_in_node_13 <= 1.1
 R3895: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_13 <= 1.1
 R3896: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.4 node_6_considers_var_3 - x136_is_in_node_13 >= -1
 R3897: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x137_is_in_node_13 <= 1.1
 R3898: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_13 <= 1.1
 R3899: - split_value_node_6 + 6.4 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.5 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x137_is_in_node_13 >= -1
 R3900: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x138_is_in_node_13 <= 1.1
 R3901: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_13 <= 1.1
 R3902: - split_value_node_6 + 6 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 4.8 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x138_is_in_node_13 >= -1
 R3903: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x139_is_in_node_13 <= 1.1
 R3904: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_13 <= 1.1
 R3905: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.4 node_6_considers_var_2
   + 2.1 node_6_considers_var_3 - x139_is_in_node_13 >= -1
 R3906: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x140_is_in_node_13 <= 1.1
 R3907: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_13 <= 1.1
 R3908: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.6 node_6_considers_var_2
   + 2.4 node_6_considers_var_3 - x140_is_in_node_13 >= -1
 R3909: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x141_is_in_node_13 <= 1.1
 R3910: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_13 <= 1.1
 R3911: - split_value_node_6 + 6.9 node_6_considers_var_0
   + 3.1 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x141_is_in_node_13 >= -1
 R3912: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x142_is_in_node_13 <= 1.1
 R3913: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_13 <= 1.1
 R3914: - split_value_node_6 + 5.8 node_6_considers_var_0
   + 2.7 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x142_is_in_node_13 >= -1
 R3915: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x143_is_in_node_13 <= 1.1
 R3916: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_13 <= 1.1
 R3917: - split_value_node_6 + 6.8 node_6_considers_var_0
   + 3.2 node_6_considers_var_1 + 5.9 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x143_is_in_node_13 >= -1
 R3918: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x144_is_in_node_13 <= 1.1
 R3919: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_13 <= 1.1
 R3920: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3.3 node_6_considers_var_1 + 5.7 node_6_considers_var_2
   + 2.5 node_6_considers_var_3 - x144_is_in_node_13 >= -1
 R3921: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x145_is_in_node_13 <= 1.1
 R3922: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_13 <= 1.1
 R3923: - split_value_node_6 + 6.7 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.2 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x145_is_in_node_13 >= -1
 R3924: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x146_is_in_node_13 <= 1.1
 R3925: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_13 <= 1.1
 R3926: - split_value_node_6 + 6.3 node_6_considers_var_0
   + 2.5 node_6_considers_var_1 + 5 node_6_considers_var_2
   + 1.9 node_6_considers_var_3 - x146_is_in_node_13 >= -1
 R3927: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x147_is_in_node_13 <= 1.1
 R3928: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_13 <= 1.1
 R3929: - split_value_node_6 + 6.5 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.2 node_6_considers_var_2
   + 2 node_6_considers_var_3 - x147_is_in_node_13 >= -1
 R3930: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x148_is_in_node_13 <= 1.1
 R3931: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_13 <= 1.1
 R3932: - split_value_node_6 + 6.2 node_6_considers_var_0
   + 3.4 node_6_considers_var_1 + 5.4 node_6_considers_var_2
   + 2.3 node_6_considers_var_3 - x148_is_in_node_13 >= -1
 R3933: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x149_is_in_node_13 <= 1.1
 R3934: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_13 <= 1.1
 R3935: - split_value_node_6 + 5.9 node_6_considers_var_0
   + 3 node_6_considers_var_1 + 5.1 node_6_considers_var_2
   + 1.8 node_6_considers_var_3 - x149_is_in_node_13 >= -1
 R3936: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x0_is_in_node_14 <= 1.1
 R3937: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x0_is_in_node_14 <= 1.1
 R3938: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_14 <= 1.1
 R3939: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x1_is_in_node_14 <= 1.1
 R3940: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x1_is_in_node_14 <= 1.1
 R3941: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_14 <= 1.1
 R3942: - split_value_node_7 + 4.8 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x2_is_in_node_14 <= 1.1
 R3943: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x2_is_in_node_14 <= 1.1
 R3944: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_14 <= 1.1
 R3945: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x3_is_in_node_14 <= 1.1
 R3946: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x3_is_in_node_14 <= 1.1
 R3947: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_14 <= 1.1
 R3948: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.7 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x4_is_in_node_14 <= 1.1
 R3949: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x4_is_in_node_14 <= 1.1
 R3950: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_14 <= 1.1
 R3951: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 4 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x5_is_in_node_14 <= 1.1
 R3952: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x5_is_in_node_14 <= 1.1
 R3953: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_14 <= 1.1
 R3954: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x6_is_in_node_14 <= 1.1
 R3955: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x6_is_in_node_14 <= 1.1
 R3956: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_14 <= 1.1
 R3957: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x7_is_in_node_14 <= 1.1
 R3958: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x7_is_in_node_14 <= 1.1
 R3959: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_14 <= 1.1
 R3960: - split_value_node_7 + 4.5 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x8_is_in_node_14 <= 1.1
 R3961: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x8_is_in_node_14 <= 1.1
 R3962: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_14 <= 1.1
 R3963: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x9_is_in_node_14 <= 1.1
 R3964: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x9_is_in_node_14 <= 1.1
 R3965: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_14 <= 1.1
 R3966: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.8 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x10_is_in_node_14 <= 1.1
 R3967: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x10_is_in_node_14 <= 1.1
 R3968: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_14 <= 1.1
 R3969: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x11_is_in_node_14 <= 1.1
 R3970: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x11_is_in_node_14 <= 1.1
 R3971: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_14 <= 1.1
 R3972: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x12_is_in_node_14 <= 1.1
 R3973: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x12_is_in_node_14 <= 1.1
 R3974: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_14 <= 1.1
 R3975: - split_value_node_7 + 4.4 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.2 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x13_is_in_node_14 <= 1.1
 R3976: - split_value_node_3 + 4.4 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.2 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x13_is_in_node_14 <= 1.1
 R3977: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_14 <= 1.1
 R3978: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 4.1 node_7_considers_var_1 + 1.3 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x14_is_in_node_14 <= 1.1
 R3979: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 4.1 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x14_is_in_node_14 <= 1.1
 R3980: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_14 <= 1.1
 R3981: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 4.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x15_is_in_node_14 <= 1.1
 R3982: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 4.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x15_is_in_node_14 <= 1.1
 R3983: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_14 <= 1.1
 R3984: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 4 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x16_is_in_node_14 <= 1.1
 R3985: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x16_is_in_node_14 <= 1.1
 R3986: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_14 <= 1.1
 R3987: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x17_is_in_node_14 <= 1.1
 R3988: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x17_is_in_node_14 <= 1.1
 R3989: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_14 <= 1.1
 R3990: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x18_is_in_node_14 <= 1.1
 R3991: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x18_is_in_node_14 <= 1.1
 R3992: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_14 <= 1.1
 R3993: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x19_is_in_node_14 <= 1.1
 R3994: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x19_is_in_node_14 <= 1.1
 R3995: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_14 <= 1.1
 R3996: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x20_is_in_node_14 <= 1.1
 R3997: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x20_is_in_node_14 <= 1.1
 R3998: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_14 <= 1.1
 R3999: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.8 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x21_is_in_node_14 <= 1.1
 R4000: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x21_is_in_node_14 <= 1.1
 R4001: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_14 <= 1.1
 R4002: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.7 node_7_considers_var_1 + 1.1 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x22_is_in_node_14 <= 1.1
 R4003: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.1 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x22_is_in_node_14 <= 1.1
 R4004: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_14 <= 1.1
 R4005: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.6 node_7_considers_var_3 + 1.1 x23_is_in_node_14 <= 1.1
 R4006: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.6 node_3_considers_var_3 + 1.1 x23_is_in_node_14 <= 1.1
 R4007: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_14 <= 1.1
 R4008: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 2 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x24_is_in_node_14 <= 1.1
 R4009: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x24_is_in_node_14 <= 1.1
 R4010: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_14 <= 1.1
 R4011: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x25_is_in_node_14 <= 1.1
 R4012: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x25_is_in_node_14 <= 1.1
 R4013: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_14 <= 1.1
 R4014: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x26_is_in_node_14 <= 1.1
 R4015: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x26_is_in_node_14 <= 1.1
 R4016: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_14 <= 1.1
 R4017: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x27_is_in_node_14 <= 1.1
 R4018: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x27_is_in_node_14 <= 1.1
 R4019: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_14 <= 1.1
 R4020: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x28_is_in_node_14 <= 1.1
 R4021: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x28_is_in_node_14 <= 1.1
 R4022: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_14 <= 1.1
 R4023: - split_value_node_7 + 4.8 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x29_is_in_node_14 <= 1.1
 R4024: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x29_is_in_node_14 <= 1.1
 R4025: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_14 <= 1.1
 R4026: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x30_is_in_node_14 <= 1.1
 R4027: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x30_is_in_node_14 <= 1.1
 R4028: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_14 <= 1.1
 R4029: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x31_is_in_node_14 <= 1.1
 R4030: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x31_is_in_node_14 <= 1.1
 R4031: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_14 <= 1.1
 R4032: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 4.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x32_is_in_node_14 <= 1.1
 R4033: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 4.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x32_is_in_node_14 <= 1.1
 R4034: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_14 <= 1.1
 R4035: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 4.3 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x33_is_in_node_14 <= 1.1
 R4036: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 4.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x33_is_in_node_14 <= 1.1
 R4037: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_14 <= 1.1
 R4038: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x34_is_in_node_14 <= 1.1
 R4039: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x34_is_in_node_14 <= 1.1
 R4040: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_14 <= 1.1
 R4041: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.3 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x35_is_in_node_14 <= 1.1
 R4042: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x35_is_in_node_14 <= 1.1
 R4043: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_14 <= 1.1
 R4044: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x36_is_in_node_14 <= 1.1
 R4045: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x36_is_in_node_14 <= 1.1
 R4046: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_14 <= 1.1
 R4047: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x37_is_in_node_14 <= 1.1
 R4048: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x37_is_in_node_14 <= 1.1
 R4049: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_14 <= 1.1
 R4050: - split_value_node_7 + 4.5 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x38_is_in_node_14 <= 1.1
 R4051: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x38_is_in_node_14 <= 1.1
 R4052: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_14 <= 1.1
 R4053: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x39_is_in_node_14 <= 1.1
 R4054: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x39_is_in_node_14 <= 1.1
 R4055: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_14 <= 1.1
 R4056: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x40_is_in_node_14 <= 1.1
 R4057: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x40_is_in_node_14 <= 1.1
 R4058: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_14 <= 1.1
 R4059: - split_value_node_7 + 4.6 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x41_is_in_node_14 <= 1.1
 R4060: - split_value_node_3 + 4.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x41_is_in_node_14 <= 1.1
 R4061: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_14 <= 1.1
 R4062: - split_value_node_7 + 4.5 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x42_is_in_node_14 <= 1.1
 R4063: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x42_is_in_node_14 <= 1.1
 R4064: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_14 <= 1.1
 R4065: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.7 node_7_considers_var_3 + 1.1 x43_is_in_node_14 <= 1.1
 R4066: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.7 node_3_considers_var_3 + 1.1 x43_is_in_node_14 <= 1.1
 R4067: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_14 <= 1.1
 R4068: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 2 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x44_is_in_node_14 <= 1.1
 R4069: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x44_is_in_node_14 <= 1.1
 R4070: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_14 <= 1.1
 R4071: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x45_is_in_node_14 <= 1.1
 R4072: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x45_is_in_node_14 <= 1.1
 R4073: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_14 <= 1.1
 R4074: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x46_is_in_node_14 <= 1.1
 R4075: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x46_is_in_node_14 <= 1.1
 R4076: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_14 <= 1.1
 R4077: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x47_is_in_node_14 <= 1.1
 R4078: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x47_is_in_node_14 <= 1.1
 R4079: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_14 <= 1.1
 R4080: - split_value_node_7 + 5.4 node_7_considers_var_0
   + 3.8 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x48_is_in_node_14 <= 1.1
 R4081: - split_value_node_3 + 5.4 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x48_is_in_node_14 <= 1.1
 R4082: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_14 <= 1.1
 R4083: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x49_is_in_node_14 <= 1.1
 R4084: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x49_is_in_node_14 <= 1.1
 R4085: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_14 <= 1.1
 R4086: - split_value_node_7 + 7.1 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x50_is_in_node_14 <= 1.1
 R4087: - split_value_node_3 + 7.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x50_is_in_node_14 <= 1.1
 R4088: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_14 <= 1.1
 R4089: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x51_is_in_node_14 <= 1.1
 R4090: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x51_is_in_node_14 <= 1.1
 R4091: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_14 <= 1.1
 R4092: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x52_is_in_node_14 <= 1.1
 R4093: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x52_is_in_node_14 <= 1.1
 R4094: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_14 <= 1.1
 R4095: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x53_is_in_node_14 <= 1.1
 R4096: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x53_is_in_node_14 <= 1.1
 R4097: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_14 <= 1.1
 R4098: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.7 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x54_is_in_node_14 <= 1.1
 R4099: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x54_is_in_node_14 <= 1.1
 R4100: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_14 <= 1.1
 R4101: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x55_is_in_node_14 <= 1.1
 R4102: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x55_is_in_node_14 <= 1.1
 R4103: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_14 <= 1.1
 R4104: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x56_is_in_node_14 <= 1.1
 R4105: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x56_is_in_node_14 <= 1.1
 R4106: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_14 <= 1.1
 R4107: - split_value_node_7 + 5 node_7_considers_var_0
   + 2.5 node_7_considers_var_1 + 3.4 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x57_is_in_node_14 <= 1.1
 R4108: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x57_is_in_node_14 <= 1.1
 R4109: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_14 <= 1.1
 R4110: - split_value_node_7 + 6.7 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.7 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x58_is_in_node_14 <= 1.1
 R4111: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x58_is_in_node_14 <= 1.1
 R4112: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_14 <= 1.1
 R4113: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x59_is_in_node_14 <= 1.1
 R4114: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x59_is_in_node_14 <= 1.1
 R4115: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_14 <= 1.1
 R4116: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 2.1 node_7_considers_var_1 + 3.6 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x60_is_in_node_14 <= 1.1
 R4117: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.1 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x60_is_in_node_14 <= 1.1
 R4118: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_14 <= 1.1
 R4119: - split_value_node_7 + 6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x61_is_in_node_14 <= 1.1
 R4120: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x61_is_in_node_14 <= 1.1
 R4121: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_14 <= 1.1
 R4122: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 2.3 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x62_is_in_node_14 <= 1.1
 R4123: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x62_is_in_node_14 <= 1.1
 R4124: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_14 <= 1.1
 R4125: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x63_is_in_node_14 <= 1.1
 R4126: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x63_is_in_node_14 <= 1.1
 R4127: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_14 <= 1.1
 R4128: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 3.7 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x64_is_in_node_14 <= 1.1
 R4129: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 3.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x64_is_in_node_14 <= 1.1
 R4130: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_14 <= 1.1
 R4131: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x65_is_in_node_14 <= 1.1
 R4132: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x65_is_in_node_14 <= 1.1
 R4133: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_14 <= 1.1
 R4134: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x66_is_in_node_14 <= 1.1
 R4135: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x66_is_in_node_14 <= 1.1
 R4136: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_14 <= 1.1
 R4137: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4.2 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x67_is_in_node_14 <= 1.1
 R4138: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x67_is_in_node_14 <= 1.1
 R4139: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_14 <= 1.1
 R4140: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 2.3 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x68_is_in_node_14 <= 1.1
 R4141: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x68_is_in_node_14 <= 1.1
 R4142: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_14 <= 1.1
 R4143: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 4 node_7_considers_var_2
   + 1.2 node_7_considers_var_3 + 1.1 x69_is_in_node_14 <= 1.1
 R4144: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x69_is_in_node_14 <= 1.1
 R4145: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_14 <= 1.1
 R4146: - split_value_node_7 + 6 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x70_is_in_node_14 <= 1.1
 R4147: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x70_is_in_node_14 <= 1.1
 R4148: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_14 <= 1.1
 R4149: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x71_is_in_node_14 <= 1.1
 R4150: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x71_is_in_node_14 <= 1.1
 R4151: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_14 <= 1.1
 R4152: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x72_is_in_node_14 <= 1.1
 R4153: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x72_is_in_node_14 <= 1.1
 R4154: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_14 <= 1.1
 R4155: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x73_is_in_node_14 <= 1.1
 R4156: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x73_is_in_node_14 <= 1.1
 R4157: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_14 <= 1.1
 R4158: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.4 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x74_is_in_node_14 <= 1.1
 R4159: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x74_is_in_node_14 <= 1.1
 R4160: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_14 <= 1.1
 R4161: - split_value_node_7 + 6.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x75_is_in_node_14 <= 1.1
 R4162: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x75_is_in_node_14 <= 1.1
 R4163: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_14 <= 1.1
 R4164: - split_value_node_7 + 6.9 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x76_is_in_node_14 <= 1.1
 R4165: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x76_is_in_node_14 <= 1.1
 R4166: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_14 <= 1.1
 R4167: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 1.8 node_7_considers_var_3 + 1.1 x77_is_in_node_14 <= 1.1
 R4168: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x77_is_in_node_14 <= 1.1
 R4169: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_14 <= 1.1
 R4170: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x78_is_in_node_14 <= 1.1
 R4171: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x78_is_in_node_14 <= 1.1
 R4172: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_14 <= 1.1
 R4173: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 3.6 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x79_is_in_node_14 <= 1.1
 R4174: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x79_is_in_node_14 <= 1.1
 R4175: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_14 <= 1.1
 R4176: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.5 node_7_considers_var_1 + 3.9 node_7_considers_var_2
   + 1.2 node_7_considers_var_3 + 1.1 x80_is_in_node_14 <= 1.1
 R4177: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.9 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x80_is_in_node_14 <= 1.1
 R4178: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_14 <= 1.1
 R4179: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.5 node_7_considers_var_1 + 3.8 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x81_is_in_node_14 <= 1.1
 R4180: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.8 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x81_is_in_node_14 <= 1.1
 R4181: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_14 <= 1.1
 R4182: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x82_is_in_node_14 <= 1.1
 R4183: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x82_is_in_node_14 <= 1.1
 R4184: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_14 <= 1.1
 R4185: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x83_is_in_node_14 <= 1.1
 R4186: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x83_is_in_node_14 <= 1.1
 R4187: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_14 <= 1.1
 R4188: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x84_is_in_node_14 <= 1.1
 R4189: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x84_is_in_node_14 <= 1.1
 R4190: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_14 <= 1.1
 R4191: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x85_is_in_node_14 <= 1.1
 R4192: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x85_is_in_node_14 <= 1.1
 R4193: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_14 <= 1.1
 R4194: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x86_is_in_node_14 <= 1.1
 R4195: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x86_is_in_node_14 <= 1.1
 R4196: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_14 <= 1.1
 R4197: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x87_is_in_node_14 <= 1.1
 R4198: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x87_is_in_node_14 <= 1.1
 R4199: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_14 <= 1.1
 R4200: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.2 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x88_is_in_node_14 <= 1.1
 R4201: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x88_is_in_node_14 <= 1.1
 R4202: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_14 <= 1.1
 R4203: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x89_is_in_node_14 <= 1.1
 R4204: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x89_is_in_node_14 <= 1.1
 R4205: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_14 <= 1.1
 R4206: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x90_is_in_node_14 <= 1.1
 R4207: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x90_is_in_node_14 <= 1.1
 R4208: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_14 <= 1.1
 R4209: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.7 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x91_is_in_node_14 <= 1.1
 R4210: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x91_is_in_node_14 <= 1.1
 R4211: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_14 <= 1.1
 R4212: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x92_is_in_node_14 <= 1.1
 R4213: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x92_is_in_node_14 <= 1.1
 R4214: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_14 <= 1.1
 R4215: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 3.4 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x93_is_in_node_14 <= 1.1
 R4216: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x93_is_in_node_14 <= 1.1
 R4217: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_14 <= 1.1
 R4218: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x94_is_in_node_14 <= 1.1
 R4219: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x94_is_in_node_14 <= 1.1
 R4220: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_14 <= 1.1
 R4221: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x95_is_in_node_14 <= 1.1
 R4222: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x95_is_in_node_14 <= 1.1
 R4223: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_14 <= 1.1
 R4224: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x96_is_in_node_14 <= 1.1
 R4225: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x96_is_in_node_14 <= 1.1
 R4226: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_14 <= 1.1
 R4227: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.4 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x97_is_in_node_14 <= 1.1
 R4228: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x97_is_in_node_14 <= 1.1
 R4229: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_14 <= 1.1
 R4230: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 3.1 node_7_considers_var_2
   + 1.2 node_7_considers_var_3 + 1.1 x98_is_in_node_14 <= 1.1
 R4231: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 3.1 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x98_is_in_node_14 <= 1.1
 R4232: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_14 <= 1.1
 R4233: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.2 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x99_is_in_node_14 <= 1.1
 R4234: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x99_is_in_node_14 <= 1.1
 R4235: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_14 <= 1.1
 R4236: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 6.1 node_7_considers_var_2
   + 2.6 node_7_considers_var_3 + 1.1 x100_is_in_node_14 <= 1.1
 R4237: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x100_is_in_node_14 <= 1.1
 R4238: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_14 <= 1.1
 R4239: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x101_is_in_node_14 <= 1.1
 R4240: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x101_is_in_node_14 <= 1.1
 R4241: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_14 <= 1.1
 R4242: - split_value_node_7 + 7.2 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 6 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x102_is_in_node_14 <= 1.1
 R4243: - split_value_node_3 + 7.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x102_is_in_node_14 <= 1.1
 R4244: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_14 <= 1.1
 R4245: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x103_is_in_node_14 <= 1.1
 R4246: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x103_is_in_node_14 <= 1.1
 R4247: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_14 <= 1.1
 R4248: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.9 node_7_considers_var_2
   + 2.3 node_7_considers_var_3 + 1.1 x104_is_in_node_14 <= 1.1
 R4249: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x104_is_in_node_14 <= 1.1
 R4250: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_14 <= 1.1
 R4251: - split_value_node_7 + 7.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 6.7 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x105_is_in_node_14 <= 1.1
 R4252: - split_value_node_3 + 7.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x105_is_in_node_14 <= 1.1
 R4253: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_14 <= 1.1
 R4254: - split_value_node_7 + 5 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.8 node_7_considers_var_3 + 1.1 x106_is_in_node_14 <= 1.1
 R4255: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x106_is_in_node_14 <= 1.1
 R4256: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_14 <= 1.1
 R4257: - split_value_node_7 + 7.4 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 6.4 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x107_is_in_node_14 <= 1.1
 R4258: - split_value_node_3 + 7.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 6.4 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x107_is_in_node_14 <= 1.1
 R4259: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_14 <= 1.1
 R4260: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x108_is_in_node_14 <= 1.1
 R4261: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x108_is_in_node_14 <= 1.1
 R4262: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_14 <= 1.1
 R4263: - split_value_node_7 + 7.3 node_7_considers_var_0
   + 3.7 node_7_considers_var_1 + 6.2 node_7_considers_var_2
   + 2.6 node_7_considers_var_3 + 1.1 x109_is_in_node_14 <= 1.1
 R4264: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x109_is_in_node_14 <= 1.1
 R4265: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_14 <= 1.1
 R4266: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x110_is_in_node_14 <= 1.1
 R4267: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x110_is_in_node_14 <= 1.1
 R4268: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_14 <= 1.1
 R4269: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.4 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x111_is_in_node_14 <= 1.1
 R4270: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x111_is_in_node_14 <= 1.1
 R4271: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_14 <= 1.1
 R4272: - split_value_node_7 + 6.9 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.6 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x112_is_in_node_14 <= 1.1
 R4273: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x112_is_in_node_14 <= 1.1
 R4274: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_14 <= 1.1
 R4275: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x113_is_in_node_14 <= 1.1
 R4276: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x113_is_in_node_14 <= 1.1
 R4277: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_14 <= 1.1
 R4278: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2.5 node_7_considers_var_3 + 1.1 x114_is_in_node_14 <= 1.1
 R4279: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x114_is_in_node_14 <= 1.1
 R4280: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_14 <= 1.1
 R4281: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 5.4 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x115_is_in_node_14 <= 1.1
 R4282: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x115_is_in_node_14 <= 1.1
 R4283: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_14 <= 1.1
 R4284: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.6 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x116_is_in_node_14 <= 1.1
 R4285: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x116_is_in_node_14 <= 1.1
 R4286: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_14 <= 1.1
 R4287: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 6.8 node_7_considers_var_2
   + 2.3 node_7_considers_var_3 + 1.1 x117_is_in_node_14 <= 1.1
 R4288: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x117_is_in_node_14 <= 1.1
 R4289: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_14 <= 1.1
 R4290: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 7 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x118_is_in_node_14 <= 1.1
 R4291: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 7 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x118_is_in_node_14 <= 1.1
 R4292: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_14 <= 1.1
 R4293: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 2.3 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x119_is_in_node_14 <= 1.1
 R4294: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x119_is_in_node_14 <= 1.1
 R4295: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_14 <= 1.1
 R4296: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 5.8 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x120_is_in_node_14 <= 1.1
 R4297: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x120_is_in_node_14 <= 1.1
 R4298: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_14 <= 1.1
 R4299: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x121_is_in_node_14 <= 1.1
 R4300: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x121_is_in_node_14 <= 1.1
 R4301: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_14 <= 1.1
 R4302: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 6.8 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x122_is_in_node_14 <= 1.1
 R4303: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x122_is_in_node_14 <= 1.1
 R4304: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_14 <= 1.1
 R4305: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x123_is_in_node_14 <= 1.1
 R4306: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x123_is_in_node_14 <= 1.1
 R4307: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_14 <= 1.1
 R4308: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 5.8 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x124_is_in_node_14 <= 1.1
 R4309: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x124_is_in_node_14 <= 1.1
 R4310: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_14 <= 1.1
 R4311: - split_value_node_7 + 7.3 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 6.1 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x125_is_in_node_14 <= 1.1
 R4312: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x125_is_in_node_14 <= 1.1
 R4313: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_14 <= 1.1
 R4314: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x126_is_in_node_14 <= 1.1
 R4315: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x126_is_in_node_14 <= 1.1
 R4316: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_14 <= 1.1
 R4317: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x127_is_in_node_14 <= 1.1
 R4318: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x127_is_in_node_14 <= 1.1
 R4319: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_14 <= 1.1
 R4320: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x128_is_in_node_14 <= 1.1
 R4321: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x128_is_in_node_14 <= 1.1
 R4322: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_14 <= 1.1
 R4323: - split_value_node_7 + 7.3 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.9 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x129_is_in_node_14 <= 1.1
 R4324: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x129_is_in_node_14 <= 1.1
 R4325: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_14 <= 1.1
 R4326: - split_value_node_7 + 7.5 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 6.2 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x130_is_in_node_14 <= 1.1
 R4327: - split_value_node_3 + 7.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x130_is_in_node_14 <= 1.1
 R4328: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_14 <= 1.1
 R4329: - split_value_node_7 + 8 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 6.5 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x131_is_in_node_14 <= 1.1
 R4330: - split_value_node_3 + 8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x131_is_in_node_14 <= 1.1
 R4331: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_14 <= 1.1
 R4332: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.3 node_7_considers_var_3 + 1.1 x132_is_in_node_14 <= 1.1
 R4333: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x132_is_in_node_14 <= 1.1
 R4334: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_14 <= 1.1
 R4335: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x133_is_in_node_14 <= 1.1
 R4336: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x133_is_in_node_14 <= 1.1
 R4337: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_14 <= 1.1
 R4338: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x134_is_in_node_14 <= 1.1
 R4339: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x134_is_in_node_14 <= 1.1
 R4340: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_14 <= 1.1
 R4341: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 6.2 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x135_is_in_node_14 <= 1.1
 R4342: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x135_is_in_node_14 <= 1.1
 R4343: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_14 <= 1.1
 R4344: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.5 node_7_considers_var_3 + 1.1 x136_is_in_node_14 <= 1.1
 R4345: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x136_is_in_node_14 <= 1.1
 R4346: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_14 <= 1.1
 R4347: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.6 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x137_is_in_node_14 <= 1.1
 R4348: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x137_is_in_node_14 <= 1.1
 R4349: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_14 <= 1.1
 R4350: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x138_is_in_node_14 <= 1.1
 R4351: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x138_is_in_node_14 <= 1.1
 R4352: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_14 <= 1.1
 R4353: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.5 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x139_is_in_node_14 <= 1.1
 R4354: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x139_is_in_node_14 <= 1.1
 R4355: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_14 <= 1.1
 R4356: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.5 node_7_considers_var_3 + 1.1 x140_is_in_node_14 <= 1.1
 R4357: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x140_is_in_node_14 <= 1.1
 R4358: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_14 <= 1.1
 R4359: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x141_is_in_node_14 <= 1.1
 R4360: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x141_is_in_node_14 <= 1.1
 R4361: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_14 <= 1.1
 R4362: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x142_is_in_node_14 <= 1.1
 R4363: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x142_is_in_node_14 <= 1.1
 R4364: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_14 <= 1.1
 R4365: - split_value_node_7 + 6.9 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 6 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x143_is_in_node_14 <= 1.1
 R4366: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x143_is_in_node_14 <= 1.1
 R4367: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_14 <= 1.1
 R4368: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 5.8 node_7_considers_var_2
   + 2.6 node_7_considers_var_3 + 1.1 x144_is_in_node_14 <= 1.1
 R4369: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x144_is_in_node_14 <= 1.1
 R4370: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_14 <= 1.1
 R4371: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.3 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x145_is_in_node_14 <= 1.1
 R4372: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x145_is_in_node_14 <= 1.1
 R4373: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_14 <= 1.1
 R4374: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x146_is_in_node_14 <= 1.1
 R4375: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x146_is_in_node_14 <= 1.1
 R4376: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_14 <= 1.1
 R4377: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.3 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x147_is_in_node_14 <= 1.1
 R4378: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x147_is_in_node_14 <= 1.1
 R4379: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_14 <= 1.1
 R4380: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 5.5 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x148_is_in_node_14 <= 1.1
 R4381: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x148_is_in_node_14 <= 1.1
 R4382: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_14 <= 1.1
 R4383: - split_value_node_7 + 6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x149_is_in_node_14 <= 1.1
 R4384: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x149_is_in_node_14 <= 1.1
 R4385: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_14 <= 1.1
 R4386: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x0_is_in_node_15 <= 1.1
 R4387: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x0_is_in_node_15 <= 1.1
 R4388: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x0_is_in_node_15 <= 1.1
 R4389: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x1_is_in_node_15 <= 1.1
 R4390: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x1_is_in_node_15 <= 1.1
 R4391: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x1_is_in_node_15 <= 1.1
 R4392: - split_value_node_7 + 4.8 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x2_is_in_node_15 <= 1.1
 R4393: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x2_is_in_node_15 <= 1.1
 R4394: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x2_is_in_node_15 <= 1.1
 R4395: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x3_is_in_node_15 <= 1.1
 R4396: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x3_is_in_node_15 <= 1.1
 R4397: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x3_is_in_node_15 <= 1.1
 R4398: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.7 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x4_is_in_node_15 <= 1.1
 R4399: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x4_is_in_node_15 <= 1.1
 R4400: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x4_is_in_node_15 <= 1.1
 R4401: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 4 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x5_is_in_node_15 <= 1.1
 R4402: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x5_is_in_node_15 <= 1.1
 R4403: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x5_is_in_node_15 <= 1.1
 R4404: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x6_is_in_node_15 <= 1.1
 R4405: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x6_is_in_node_15 <= 1.1
 R4406: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x6_is_in_node_15 <= 1.1
 R4407: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x7_is_in_node_15 <= 1.1
 R4408: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x7_is_in_node_15 <= 1.1
 R4409: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x7_is_in_node_15 <= 1.1
 R4410: - split_value_node_7 + 4.5 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x8_is_in_node_15 <= 1.1
 R4411: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x8_is_in_node_15 <= 1.1
 R4412: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x8_is_in_node_15 <= 1.1
 R4413: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x9_is_in_node_15 <= 1.1
 R4414: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x9_is_in_node_15 <= 1.1
 R4415: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x9_is_in_node_15 <= 1.1
 R4416: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.8 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x10_is_in_node_15 <= 1.1
 R4417: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x10_is_in_node_15 <= 1.1
 R4418: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x10_is_in_node_15 <= 1.1
 R4419: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x11_is_in_node_15 <= 1.1
 R4420: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x11_is_in_node_15 <= 1.1
 R4421: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x11_is_in_node_15 <= 1.1
 R4422: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x12_is_in_node_15 <= 1.1
 R4423: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x12_is_in_node_15 <= 1.1
 R4424: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x12_is_in_node_15 <= 1.1
 R4425: - split_value_node_7 + 4.4 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.2 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x13_is_in_node_15 <= 1.1
 R4426: - split_value_node_3 + 4.4 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.2 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x13_is_in_node_15 <= 1.1
 R4427: - split_value_node_1 + 4.4 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.2 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x13_is_in_node_15 <= 1.1
 R4428: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 4.1 node_7_considers_var_1 + 1.3 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x14_is_in_node_15 <= 1.1
 R4429: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 4.1 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x14_is_in_node_15 <= 1.1
 R4430: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 4.1 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x14_is_in_node_15 <= 1.1
 R4431: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 4.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x15_is_in_node_15 <= 1.1
 R4432: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 4.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x15_is_in_node_15 <= 1.1
 R4433: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 4.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x15_is_in_node_15 <= 1.1
 R4434: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 4 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x16_is_in_node_15 <= 1.1
 R4435: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x16_is_in_node_15 <= 1.1
 R4436: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x16_is_in_node_15 <= 1.1
 R4437: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x17_is_in_node_15 <= 1.1
 R4438: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x17_is_in_node_15 <= 1.1
 R4439: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x17_is_in_node_15 <= 1.1
 R4440: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x18_is_in_node_15 <= 1.1
 R4441: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x18_is_in_node_15 <= 1.1
 R4442: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x18_is_in_node_15 <= 1.1
 R4443: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x19_is_in_node_15 <= 1.1
 R4444: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x19_is_in_node_15 <= 1.1
 R4445: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x19_is_in_node_15 <= 1.1
 R4446: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x20_is_in_node_15 <= 1.1
 R4447: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x20_is_in_node_15 <= 1.1
 R4448: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x20_is_in_node_15 <= 1.1
 R4449: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.8 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x21_is_in_node_15 <= 1.1
 R4450: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x21_is_in_node_15 <= 1.1
 R4451: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x21_is_in_node_15 <= 1.1
 R4452: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.7 node_7_considers_var_1 + 1.1 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x22_is_in_node_15 <= 1.1
 R4453: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 1.1 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x22_is_in_node_15 <= 1.1
 R4454: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 1.1 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x22_is_in_node_15 <= 1.1
 R4455: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 1.8 node_7_considers_var_2
   + 0.6 node_7_considers_var_3 + 1.1 x23_is_in_node_15 <= 1.1
 R4456: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.8 node_3_considers_var_2
   + 0.6 node_3_considers_var_3 + 1.1 x23_is_in_node_15 <= 1.1
 R4457: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.8 node_1_considers_var_2
   + 0.6 node_1_considers_var_3 + 1.1 x23_is_in_node_15 <= 1.1
 R4458: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 2 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x24_is_in_node_15 <= 1.1
 R4459: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x24_is_in_node_15 <= 1.1
 R4460: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x24_is_in_node_15 <= 1.1
 R4461: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x25_is_in_node_15 <= 1.1
 R4462: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x25_is_in_node_15 <= 1.1
 R4463: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x25_is_in_node_15 <= 1.1
 R4464: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x26_is_in_node_15 <= 1.1
 R4465: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x26_is_in_node_15 <= 1.1
 R4466: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x26_is_in_node_15 <= 1.1
 R4467: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x27_is_in_node_15 <= 1.1
 R4468: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x27_is_in_node_15 <= 1.1
 R4469: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x27_is_in_node_15 <= 1.1
 R4470: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x28_is_in_node_15 <= 1.1
 R4471: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x28_is_in_node_15 <= 1.1
 R4472: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x28_is_in_node_15 <= 1.1
 R4473: - split_value_node_7 + 4.8 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x29_is_in_node_15 <= 1.1
 R4474: - split_value_node_3 + 4.8 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x29_is_in_node_15 <= 1.1
 R4475: - split_value_node_1 + 4.8 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x29_is_in_node_15 <= 1.1
 R4476: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x30_is_in_node_15 <= 1.1
 R4477: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x30_is_in_node_15 <= 1.1
 R4478: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x30_is_in_node_15 <= 1.1
 R4479: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x31_is_in_node_15 <= 1.1
 R4480: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x31_is_in_node_15 <= 1.1
 R4481: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x31_is_in_node_15 <= 1.1
 R4482: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 4.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x32_is_in_node_15 <= 1.1
 R4483: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 4.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x32_is_in_node_15 <= 1.1
 R4484: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 4.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x32_is_in_node_15 <= 1.1
 R4485: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 4.3 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x33_is_in_node_15 <= 1.1
 R4486: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 4.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x33_is_in_node_15 <= 1.1
 R4487: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 4.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x33_is_in_node_15 <= 1.1
 R4488: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x34_is_in_node_15 <= 1.1
 R4489: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x34_is_in_node_15 <= 1.1
 R4490: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x34_is_in_node_15 <= 1.1
 R4491: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.3 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x35_is_in_node_15 <= 1.1
 R4492: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.3 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x35_is_in_node_15 <= 1.1
 R4493: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.3 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x35_is_in_node_15 <= 1.1
 R4494: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x36_is_in_node_15 <= 1.1
 R4495: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x36_is_in_node_15 <= 1.1
 R4496: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x36_is_in_node_15 <= 1.1
 R4497: - split_value_node_7 + 5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.2 node_7_considers_var_3 + 1.1 x37_is_in_node_15 <= 1.1
 R4498: - split_value_node_3 + 5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.2 node_3_considers_var_3 + 1.1 x37_is_in_node_15 <= 1.1
 R4499: - split_value_node_1 + 5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.2 node_1_considers_var_3 + 1.1 x37_is_in_node_15 <= 1.1
 R4500: - split_value_node_7 + 4.5 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x38_is_in_node_15 <= 1.1
 R4501: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x38_is_in_node_15 <= 1.1
 R4502: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x38_is_in_node_15 <= 1.1
 R4503: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x39_is_in_node_15 <= 1.1
 R4504: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x39_is_in_node_15 <= 1.1
 R4505: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x39_is_in_node_15 <= 1.1
 R4506: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x40_is_in_node_15 <= 1.1
 R4507: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x40_is_in_node_15 <= 1.1
 R4508: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x40_is_in_node_15 <= 1.1
 R4509: - split_value_node_7 + 4.6 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x41_is_in_node_15 <= 1.1
 R4510: - split_value_node_3 + 4.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x41_is_in_node_15 <= 1.1
 R4511: - split_value_node_1 + 4.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x41_is_in_node_15 <= 1.1
 R4512: - split_value_node_7 + 4.5 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.4 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x42_is_in_node_15 <= 1.1
 R4513: - split_value_node_3 + 4.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.4 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x42_is_in_node_15 <= 1.1
 R4514: - split_value_node_1 + 4.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.4 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x42_is_in_node_15 <= 1.1
 R4515: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.6 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.7 node_7_considers_var_3 + 1.1 x43_is_in_node_15 <= 1.1
 R4516: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.6 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.7 node_3_considers_var_3 + 1.1 x43_is_in_node_15 <= 1.1
 R4517: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.6 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.7 node_1_considers_var_3 + 1.1 x43_is_in_node_15 <= 1.1
 R4518: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 2 node_7_considers_var_2
   + 0.5 node_7_considers_var_3 + 1.1 x44_is_in_node_15 <= 1.1
 R4519: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 2 node_3_considers_var_2
   + 0.5 node_3_considers_var_3 + 1.1 x44_is_in_node_15 <= 1.1
 R4520: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 2 node_1_considers_var_2
   + 0.5 node_1_considers_var_3 + 1.1 x44_is_in_node_15 <= 1.1
 R4521: - split_value_node_7 + 4.9 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.4 node_7_considers_var_3 + 1.1 x45_is_in_node_15 <= 1.1
 R4522: - split_value_node_3 + 4.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.4 node_3_considers_var_3 + 1.1 x45_is_in_node_15 <= 1.1
 R4523: - split_value_node_1 + 4.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.4 node_1_considers_var_3 + 1.1 x45_is_in_node_15 <= 1.1
 R4524: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 1.7 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x46_is_in_node_15 <= 1.1
 R4525: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 1.7 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x46_is_in_node_15 <= 1.1
 R4526: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 1.7 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x46_is_in_node_15 <= 1.1
 R4527: - split_value_node_7 + 4.7 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x47_is_in_node_15 <= 1.1
 R4528: - split_value_node_3 + 4.7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x47_is_in_node_15 <= 1.1
 R4529: - split_value_node_1 + 4.7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x47_is_in_node_15 <= 1.1
 R4530: - split_value_node_7 + 5.4 node_7_considers_var_0
   + 3.8 node_7_considers_var_1 + 1.6 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x48_is_in_node_15 <= 1.1
 R4531: - split_value_node_3 + 5.4 node_3_considers_var_0
   + 3.8 node_3_considers_var_1 + 1.6 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x48_is_in_node_15 <= 1.1
 R4532: - split_value_node_1 + 5.4 node_1_considers_var_0
   + 3.8 node_1_considers_var_1 + 1.6 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x48_is_in_node_15 <= 1.1
 R4533: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 1.5 node_7_considers_var_2
   + 0.3 node_7_considers_var_3 + 1.1 x49_is_in_node_15 <= 1.1
 R4534: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 1.5 node_3_considers_var_2
   + 0.3 node_3_considers_var_3 + 1.1 x49_is_in_node_15 <= 1.1
 R4535: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 1.5 node_1_considers_var_2
   + 0.3 node_1_considers_var_3 + 1.1 x49_is_in_node_15 <= 1.1
 R4536: - split_value_node_7 + 7.1 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x50_is_in_node_15 <= 1.1
 R4537: - split_value_node_3 + 7.1 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x50_is_in_node_15 <= 1.1
 R4538: - split_value_node_1 + 7.1 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x50_is_in_node_15 <= 1.1
 R4539: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x51_is_in_node_15 <= 1.1
 R4540: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x51_is_in_node_15 <= 1.1
 R4541: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x51_is_in_node_15 <= 1.1
 R4542: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x52_is_in_node_15 <= 1.1
 R4543: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x52_is_in_node_15 <= 1.1
 R4544: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x52_is_in_node_15 <= 1.1
 R4545: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x53_is_in_node_15 <= 1.1
 R4546: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x53_is_in_node_15 <= 1.1
 R4547: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x53_is_in_node_15 <= 1.1
 R4548: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.7 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x54_is_in_node_15 <= 1.1
 R4549: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x54_is_in_node_15 <= 1.1
 R4550: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x54_is_in_node_15 <= 1.1
 R4551: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x55_is_in_node_15 <= 1.1
 R4552: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x55_is_in_node_15 <= 1.1
 R4553: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x55_is_in_node_15 <= 1.1
 R4554: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x56_is_in_node_15 <= 1.1
 R4555: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x56_is_in_node_15 <= 1.1
 R4556: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x56_is_in_node_15 <= 1.1
 R4557: - split_value_node_7 + 5 node_7_considers_var_0
   + 2.5 node_7_considers_var_1 + 3.4 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x57_is_in_node_15 <= 1.1
 R4558: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x57_is_in_node_15 <= 1.1
 R4559: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x57_is_in_node_15 <= 1.1
 R4560: - split_value_node_7 + 6.7 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.7 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x58_is_in_node_15 <= 1.1
 R4561: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x58_is_in_node_15 <= 1.1
 R4562: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x58_is_in_node_15 <= 1.1
 R4563: - split_value_node_7 + 5.3 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x59_is_in_node_15 <= 1.1
 R4564: - split_value_node_3 + 5.3 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x59_is_in_node_15 <= 1.1
 R4565: - split_value_node_1 + 5.3 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x59_is_in_node_15 <= 1.1
 R4566: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 2.1 node_7_considers_var_1 + 3.6 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x60_is_in_node_15 <= 1.1
 R4567: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.1 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x60_is_in_node_15 <= 1.1
 R4568: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.1 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x60_is_in_node_15 <= 1.1
 R4569: - split_value_node_7 + 6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x61_is_in_node_15 <= 1.1
 R4570: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x61_is_in_node_15 <= 1.1
 R4571: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x61_is_in_node_15 <= 1.1
 R4572: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 2.3 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x62_is_in_node_15 <= 1.1
 R4573: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x62_is_in_node_15 <= 1.1
 R4574: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x62_is_in_node_15 <= 1.1
 R4575: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x63_is_in_node_15 <= 1.1
 R4576: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x63_is_in_node_15 <= 1.1
 R4577: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x63_is_in_node_15 <= 1.1
 R4578: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 3.7 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x64_is_in_node_15 <= 1.1
 R4579: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 3.7 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x64_is_in_node_15 <= 1.1
 R4580: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 3.7 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x64_is_in_node_15 <= 1.1
 R4581: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x65_is_in_node_15 <= 1.1
 R4582: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x65_is_in_node_15 <= 1.1
 R4583: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x65_is_in_node_15 <= 1.1
 R4584: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x66_is_in_node_15 <= 1.1
 R4585: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x66_is_in_node_15 <= 1.1
 R4586: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x66_is_in_node_15 <= 1.1
 R4587: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4.2 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x67_is_in_node_15 <= 1.1
 R4588: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x67_is_in_node_15 <= 1.1
 R4589: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x67_is_in_node_15 <= 1.1
 R4590: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 2.3 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x68_is_in_node_15 <= 1.1
 R4591: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x68_is_in_node_15 <= 1.1
 R4592: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x68_is_in_node_15 <= 1.1
 R4593: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 4 node_7_considers_var_2
   + 1.2 node_7_considers_var_3 + 1.1 x69_is_in_node_15 <= 1.1
 R4594: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x69_is_in_node_15 <= 1.1
 R4595: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x69_is_in_node_15 <= 1.1
 R4596: - split_value_node_7 + 6 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x70_is_in_node_15 <= 1.1
 R4597: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x70_is_in_node_15 <= 1.1
 R4598: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x70_is_in_node_15 <= 1.1
 R4599: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x71_is_in_node_15 <= 1.1
 R4600: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x71_is_in_node_15 <= 1.1
 R4601: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x71_is_in_node_15 <= 1.1
 R4602: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x72_is_in_node_15 <= 1.1
 R4603: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x72_is_in_node_15 <= 1.1
 R4604: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x72_is_in_node_15 <= 1.1
 R4605: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x73_is_in_node_15 <= 1.1
 R4606: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x73_is_in_node_15 <= 1.1
 R4607: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x73_is_in_node_15 <= 1.1
 R4608: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.4 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x74_is_in_node_15 <= 1.1
 R4609: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x74_is_in_node_15 <= 1.1
 R4610: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x74_is_in_node_15 <= 1.1
 R4611: - split_value_node_7 + 6.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x75_is_in_node_15 <= 1.1
 R4612: - split_value_node_3 + 6.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x75_is_in_node_15 <= 1.1
 R4613: - split_value_node_1 + 6.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x75_is_in_node_15 <= 1.1
 R4614: - split_value_node_7 + 6.9 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x76_is_in_node_15 <= 1.1
 R4615: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x76_is_in_node_15 <= 1.1
 R4616: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x76_is_in_node_15 <= 1.1
 R4617: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 1.8 node_7_considers_var_3 + 1.1 x77_is_in_node_15 <= 1.1
 R4618: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x77_is_in_node_15 <= 1.1
 R4619: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x77_is_in_node_15 <= 1.1
 R4620: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x78_is_in_node_15 <= 1.1
 R4621: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x78_is_in_node_15 <= 1.1
 R4622: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x78_is_in_node_15 <= 1.1
 R4623: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 3.6 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x79_is_in_node_15 <= 1.1
 R4624: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 3.6 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x79_is_in_node_15 <= 1.1
 R4625: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 3.6 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x79_is_in_node_15 <= 1.1
 R4626: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.5 node_7_considers_var_1 + 3.9 node_7_considers_var_2
   + 1.2 node_7_considers_var_3 + 1.1 x80_is_in_node_15 <= 1.1
 R4627: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.9 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x80_is_in_node_15 <= 1.1
 R4628: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.9 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x80_is_in_node_15 <= 1.1
 R4629: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.5 node_7_considers_var_1 + 3.8 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x81_is_in_node_15 <= 1.1
 R4630: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.5 node_3_considers_var_1 + 3.8 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x81_is_in_node_15 <= 1.1
 R4631: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.5 node_1_considers_var_1 + 3.8 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x81_is_in_node_15 <= 1.1
 R4632: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x82_is_in_node_15 <= 1.1
 R4633: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x82_is_in_node_15 <= 1.1
 R4634: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x82_is_in_node_15 <= 1.1
 R4635: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x83_is_in_node_15 <= 1.1
 R4636: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x83_is_in_node_15 <= 1.1
 R4637: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x83_is_in_node_15 <= 1.1
 R4638: - split_value_node_7 + 5.5 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x84_is_in_node_15 <= 1.1
 R4639: - split_value_node_3 + 5.5 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x84_is_in_node_15 <= 1.1
 R4640: - split_value_node_1 + 5.5 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x84_is_in_node_15 <= 1.1
 R4641: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x85_is_in_node_15 <= 1.1
 R4642: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x85_is_in_node_15 <= 1.1
 R4643: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x85_is_in_node_15 <= 1.1
 R4644: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 4.8 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x86_is_in_node_15 <= 1.1
 R4645: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 4.8 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x86_is_in_node_15 <= 1.1
 R4646: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 4.8 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x86_is_in_node_15 <= 1.1
 R4647: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x87_is_in_node_15 <= 1.1
 R4648: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x87_is_in_node_15 <= 1.1
 R4649: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x87_is_in_node_15 <= 1.1
 R4650: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.2 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x88_is_in_node_15 <= 1.1
 R4651: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x88_is_in_node_15 <= 1.1
 R4652: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x88_is_in_node_15 <= 1.1
 R4653: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x89_is_in_node_15 <= 1.1
 R4654: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x89_is_in_node_15 <= 1.1
 R4655: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x89_is_in_node_15 <= 1.1
 R4656: - split_value_node_7 + 5.6 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 4.5 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x90_is_in_node_15 <= 1.1
 R4657: - split_value_node_3 + 5.6 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.5 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x90_is_in_node_15 <= 1.1
 R4658: - split_value_node_1 + 5.6 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.5 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x90_is_in_node_15 <= 1.1
 R4659: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.7 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x91_is_in_node_15 <= 1.1
 R4660: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x91_is_in_node_15 <= 1.1
 R4661: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x91_is_in_node_15 <= 1.1
 R4662: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 4.1 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x92_is_in_node_15 <= 1.1
 R4663: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 4.1 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x92_is_in_node_15 <= 1.1
 R4664: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 4.1 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x92_is_in_node_15 <= 1.1
 R4665: - split_value_node_7 + 5.1 node_7_considers_var_0
   + 2.4 node_7_considers_var_1 + 3.4 node_7_considers_var_2
   + 1.1 node_7_considers_var_3 + 1.1 x93_is_in_node_15 <= 1.1
 R4666: - split_value_node_3 + 5.1 node_3_considers_var_0
   + 2.4 node_3_considers_var_1 + 3.4 node_3_considers_var_2
   + 1.1 node_3_considers_var_3 + 1.1 x93_is_in_node_15 <= 1.1
 R4667: - split_value_node_1 + 5.1 node_1_considers_var_0
   + 2.4 node_1_considers_var_1 + 3.4 node_1_considers_var_2
   + 1.1 node_1_considers_var_3 + 1.1 x93_is_in_node_15 <= 1.1
 R4668: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x94_is_in_node_15 <= 1.1
 R4669: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x94_is_in_node_15 <= 1.1
 R4670: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x94_is_in_node_15 <= 1.1
 R4671: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.3 node_7_considers_var_3 + 1.1 x95_is_in_node_15 <= 1.1
 R4672: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.3 node_3_considers_var_3 + 1.1 x95_is_in_node_15 <= 1.1
 R4673: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.3 node_1_considers_var_3 + 1.1 x95_is_in_node_15 <= 1.1
 R4674: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.3 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x96_is_in_node_15 <= 1.1
 R4675: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.3 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x96_is_in_node_15 <= 1.1
 R4676: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.3 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x96_is_in_node_15 <= 1.1
 R4677: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 4.4 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x97_is_in_node_15 <= 1.1
 R4678: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 4.4 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x97_is_in_node_15 <= 1.1
 R4679: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 4.4 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x97_is_in_node_15 <= 1.1
 R4680: - split_value_node_7 + 5.2 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 3.1 node_7_considers_var_2
   + 1.2 node_7_considers_var_3 + 1.1 x98_is_in_node_15 <= 1.1
 R4681: - split_value_node_3 + 5.2 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 3.1 node_3_considers_var_2
   + 1.2 node_3_considers_var_3 + 1.1 x98_is_in_node_15 <= 1.1
 R4682: - split_value_node_1 + 5.2 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 3.1 node_1_considers_var_2
   + 1.2 node_1_considers_var_3 + 1.1 x98_is_in_node_15 <= 1.1
 R4683: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.2 node_7_considers_var_2
   + 1.4 node_7_considers_var_3 + 1.1 x99_is_in_node_15 <= 1.1
 R4684: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.2 node_3_considers_var_2
   + 1.4 node_3_considers_var_3 + 1.1 x99_is_in_node_15 <= 1.1
 R4685: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.2 node_1_considers_var_2
   + 1.4 node_1_considers_var_3 + 1.1 x99_is_in_node_15 <= 1.1
 R4686: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 6.1 node_7_considers_var_2
   + 2.6 node_7_considers_var_3 + 1.1 x100_is_in_node_15 <= 1.1
 R4687: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x100_is_in_node_15 <= 1.1
 R4688: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x100_is_in_node_15 <= 1.1
 R4689: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x101_is_in_node_15 <= 1.1
 R4690: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x101_is_in_node_15 <= 1.1
 R4691: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x101_is_in_node_15 <= 1.1
 R4692: - split_value_node_7 + 7.2 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 6 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x102_is_in_node_15 <= 1.1
 R4693: - split_value_node_3 + 7.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x102_is_in_node_15 <= 1.1
 R4694: - split_value_node_1 + 7.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x102_is_in_node_15 <= 1.1
 R4695: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x103_is_in_node_15 <= 1.1
 R4696: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x103_is_in_node_15 <= 1.1
 R4697: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x103_is_in_node_15 <= 1.1
 R4698: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.9 node_7_considers_var_2
   + 2.3 node_7_considers_var_3 + 1.1 x104_is_in_node_15 <= 1.1
 R4699: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x104_is_in_node_15 <= 1.1
 R4700: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x104_is_in_node_15 <= 1.1
 R4701: - split_value_node_7 + 7.7 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 6.7 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x105_is_in_node_15 <= 1.1
 R4702: - split_value_node_3 + 7.7 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x105_is_in_node_15 <= 1.1
 R4703: - split_value_node_1 + 7.7 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x105_is_in_node_15 <= 1.1
 R4704: - split_value_node_7 + 5 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 4.6 node_7_considers_var_2
   + 1.8 node_7_considers_var_3 + 1.1 x106_is_in_node_15 <= 1.1
 R4705: - split_value_node_3 + 5 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 4.6 node_3_considers_var_2
   + 1.8 node_3_considers_var_3 + 1.1 x106_is_in_node_15 <= 1.1
 R4706: - split_value_node_1 + 5 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 4.6 node_1_considers_var_2
   + 1.8 node_1_considers_var_3 + 1.1 x106_is_in_node_15 <= 1.1
 R4707: - split_value_node_7 + 7.4 node_7_considers_var_0
   + 3 node_7_considers_var_1 + 6.4 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x107_is_in_node_15 <= 1.1
 R4708: - split_value_node_3 + 7.4 node_3_considers_var_0
   + 3 node_3_considers_var_1 + 6.4 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x107_is_in_node_15 <= 1.1
 R4709: - split_value_node_1 + 7.4 node_1_considers_var_0
   + 3 node_1_considers_var_1 + 6.4 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x107_is_in_node_15 <= 1.1
 R4710: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x108_is_in_node_15 <= 1.1
 R4711: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x108_is_in_node_15 <= 1.1
 R4712: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x108_is_in_node_15 <= 1.1
 R4713: - split_value_node_7 + 7.3 node_7_considers_var_0
   + 3.7 node_7_considers_var_1 + 6.2 node_7_considers_var_2
   + 2.6 node_7_considers_var_3 + 1.1 x109_is_in_node_15 <= 1.1
 R4714: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.7 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x109_is_in_node_15 <= 1.1
 R4715: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.7 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x109_is_in_node_15 <= 1.1
 R4716: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x110_is_in_node_15 <= 1.1
 R4717: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x110_is_in_node_15 <= 1.1
 R4718: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x110_is_in_node_15 <= 1.1
 R4719: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.4 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x111_is_in_node_15 <= 1.1
 R4720: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x111_is_in_node_15 <= 1.1
 R4721: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x111_is_in_node_15 <= 1.1
 R4722: - split_value_node_7 + 6.9 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.6 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x112_is_in_node_15 <= 1.1
 R4723: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x112_is_in_node_15 <= 1.1
 R4724: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x112_is_in_node_15 <= 1.1
 R4725: - split_value_node_7 + 5.8 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x113_is_in_node_15 <= 1.1
 R4726: - split_value_node_3 + 5.8 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x113_is_in_node_15 <= 1.1
 R4727: - split_value_node_1 + 5.8 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x113_is_in_node_15 <= 1.1
 R4728: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2.5 node_7_considers_var_3 + 1.1 x114_is_in_node_15 <= 1.1
 R4729: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x114_is_in_node_15 <= 1.1
 R4730: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x114_is_in_node_15 <= 1.1
 R4731: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 5.4 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x115_is_in_node_15 <= 1.1
 R4732: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.4 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x115_is_in_node_15 <= 1.1
 R4733: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.4 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x115_is_in_node_15 <= 1.1
 R4734: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.6 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x116_is_in_node_15 <= 1.1
 R4735: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x116_is_in_node_15 <= 1.1
 R4736: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x116_is_in_node_15 <= 1.1
 R4737: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 6.8 node_7_considers_var_2
   + 2.3 node_7_considers_var_3 + 1.1 x117_is_in_node_15 <= 1.1
 R4738: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x117_is_in_node_15 <= 1.1
 R4739: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x117_is_in_node_15 <= 1.1
 R4740: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 7 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x118_is_in_node_15 <= 1.1
 R4741: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 7 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x118_is_in_node_15 <= 1.1
 R4742: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 7 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x118_is_in_node_15 <= 1.1
 R4743: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 2.3 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x119_is_in_node_15 <= 1.1
 R4744: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 2.3 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x119_is_in_node_15 <= 1.1
 R4745: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 2.3 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x119_is_in_node_15 <= 1.1
 R4746: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 5.8 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x120_is_in_node_15 <= 1.1
 R4747: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x120_is_in_node_15 <= 1.1
 R4748: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x120_is_in_node_15 <= 1.1
 R4749: - split_value_node_7 + 5.7 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x121_is_in_node_15 <= 1.1
 R4750: - split_value_node_3 + 5.7 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x121_is_in_node_15 <= 1.1
 R4751: - split_value_node_1 + 5.7 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x121_is_in_node_15 <= 1.1
 R4752: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 6.8 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x122_is_in_node_15 <= 1.1
 R4753: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.8 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x122_is_in_node_15 <= 1.1
 R4754: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.8 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x122_is_in_node_15 <= 1.1
 R4755: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x123_is_in_node_15 <= 1.1
 R4756: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x123_is_in_node_15 <= 1.1
 R4757: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x123_is_in_node_15 <= 1.1
 R4758: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 5.8 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x124_is_in_node_15 <= 1.1
 R4759: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x124_is_in_node_15 <= 1.1
 R4760: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x124_is_in_node_15 <= 1.1
 R4761: - split_value_node_7 + 7.3 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 6.1 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x125_is_in_node_15 <= 1.1
 R4762: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6.1 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x125_is_in_node_15 <= 1.1
 R4763: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6.1 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x125_is_in_node_15 <= 1.1
 R4764: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x126_is_in_node_15 <= 1.1
 R4765: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x126_is_in_node_15 <= 1.1
 R4766: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x126_is_in_node_15 <= 1.1
 R4767: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x127_is_in_node_15 <= 1.1
 R4768: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x127_is_in_node_15 <= 1.1
 R4769: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x127_is_in_node_15 <= 1.1
 R4770: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x128_is_in_node_15 <= 1.1
 R4771: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x128_is_in_node_15 <= 1.1
 R4772: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x128_is_in_node_15 <= 1.1
 R4773: - split_value_node_7 + 7.3 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.9 node_7_considers_var_2
   + 1.7 node_7_considers_var_3 + 1.1 x129_is_in_node_15 <= 1.1
 R4774: - split_value_node_3 + 7.3 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.9 node_3_considers_var_2
   + 1.7 node_3_considers_var_3 + 1.1 x129_is_in_node_15 <= 1.1
 R4775: - split_value_node_1 + 7.3 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.9 node_1_considers_var_2
   + 1.7 node_1_considers_var_3 + 1.1 x129_is_in_node_15 <= 1.1
 R4776: - split_value_node_7 + 7.5 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 6.2 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x130_is_in_node_15 <= 1.1
 R4777: - split_value_node_3 + 7.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x130_is_in_node_15 <= 1.1
 R4778: - split_value_node_1 + 7.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x130_is_in_node_15 <= 1.1
 R4779: - split_value_node_7 + 8 node_7_considers_var_0
   + 3.9 node_7_considers_var_1 + 6.5 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x131_is_in_node_15 <= 1.1
 R4780: - split_value_node_3 + 8 node_3_considers_var_0
   + 3.9 node_3_considers_var_1 + 6.5 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x131_is_in_node_15 <= 1.1
 R4781: - split_value_node_1 + 8 node_1_considers_var_0
   + 3.9 node_1_considers_var_1 + 6.5 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x131_is_in_node_15 <= 1.1
 R4782: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.3 node_7_considers_var_3 + 1.1 x132_is_in_node_15 <= 1.1
 R4783: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.3 node_3_considers_var_3 + 1.1 x132_is_in_node_15 <= 1.1
 R4784: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.3 node_1_considers_var_3 + 1.1 x132_is_in_node_15 <= 1.1
 R4785: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.9 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 1.6 node_7_considers_var_3 + 1.1 x133_is_in_node_15 <= 1.1
 R4786: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.9 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.6 node_3_considers_var_3 + 1.1 x133_is_in_node_15 <= 1.1
 R4787: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.9 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.6 node_1_considers_var_3 + 1.1 x133_is_in_node_15 <= 1.1
 R4788: - split_value_node_7 + 6.2 node_7_considers_var_0
   + 2.7 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 1.5 node_7_considers_var_3 + 1.1 x134_is_in_node_15 <= 1.1
 R4789: - split_value_node_3 + 6.2 node_3_considers_var_0
   + 2.7 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 1.5 node_3_considers_var_3 + 1.1 x134_is_in_node_15 <= 1.1
 R4790: - split_value_node_1 + 6.2 node_1_considers_var_0
   + 2.7 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 1.5 node_1_considers_var_3 + 1.1 x134_is_in_node_15 <= 1.1
 R4791: - split_value_node_7 + 7.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 6.2 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x135_is_in_node_15 <= 1.1
 R4792: - split_value_node_3 + 7.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 6.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x135_is_in_node_15 <= 1.1
 R4793: - split_value_node_1 + 7.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 6.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x135_is_in_node_15 <= 1.1
 R4794: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.5 node_7_considers_var_3 + 1.1 x136_is_in_node_15 <= 1.1
 R4795: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x136_is_in_node_15 <= 1.1
 R4796: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x136_is_in_node_15 <= 1.1
 R4797: - split_value_node_7 + 6.5 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.6 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x137_is_in_node_15 <= 1.1
 R4798: - split_value_node_3 + 6.5 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.6 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x137_is_in_node_15 <= 1.1
 R4799: - split_value_node_1 + 6.5 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.6 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x137_is_in_node_15 <= 1.1
 R4800: - split_value_node_7 + 6.1 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 4.9 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x138_is_in_node_15 <= 1.1
 R4801: - split_value_node_3 + 6.1 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 4.9 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x138_is_in_node_15 <= 1.1
 R4802: - split_value_node_1 + 6.1 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 4.9 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x138_is_in_node_15 <= 1.1
 R4803: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.5 node_7_considers_var_2
   + 2.2 node_7_considers_var_3 + 1.1 x139_is_in_node_15 <= 1.1
 R4804: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.2 node_3_considers_var_3 + 1.1 x139_is_in_node_15 <= 1.1
 R4805: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.2 node_1_considers_var_3 + 1.1 x139_is_in_node_15 <= 1.1
 R4806: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.7 node_7_considers_var_2
   + 2.5 node_7_considers_var_3 + 1.1 x140_is_in_node_15 <= 1.1
 R4807: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.7 node_3_considers_var_2
   + 2.5 node_3_considers_var_3 + 1.1 x140_is_in_node_15 <= 1.1
 R4808: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.7 node_1_considers_var_2
   + 2.5 node_1_considers_var_3 + 1.1 x140_is_in_node_15 <= 1.1
 R4809: - split_value_node_7 + 7 node_7_considers_var_0
   + 3.2 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x141_is_in_node_15 <= 1.1
 R4810: - split_value_node_3 + 7 node_3_considers_var_0
   + 3.2 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x141_is_in_node_15 <= 1.1
 R4811: - split_value_node_1 + 7 node_1_considers_var_0
   + 3.2 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x141_is_in_node_15 <= 1.1
 R4812: - split_value_node_7 + 5.9 node_7_considers_var_0
   + 2.8 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x142_is_in_node_15 <= 1.1
 R4813: - split_value_node_3 + 5.9 node_3_considers_var_0
   + 2.8 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x142_is_in_node_15 <= 1.1
 R4814: - split_value_node_1 + 5.9 node_1_considers_var_0
   + 2.8 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x142_is_in_node_15 <= 1.1
 R4815: - split_value_node_7 + 6.9 node_7_considers_var_0
   + 3.3 node_7_considers_var_1 + 6 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x143_is_in_node_15 <= 1.1
 R4816: - split_value_node_3 + 6.9 node_3_considers_var_0
   + 3.3 node_3_considers_var_1 + 6 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x143_is_in_node_15 <= 1.1
 R4817: - split_value_node_1 + 6.9 node_1_considers_var_0
   + 3.3 node_1_considers_var_1 + 6 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x143_is_in_node_15 <= 1.1
 R4818: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.4 node_7_considers_var_1 + 5.8 node_7_considers_var_2
   + 2.6 node_7_considers_var_3 + 1.1 x144_is_in_node_15 <= 1.1
 R4819: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.4 node_3_considers_var_1 + 5.8 node_3_considers_var_2
   + 2.6 node_3_considers_var_3 + 1.1 x144_is_in_node_15 <= 1.1
 R4820: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.4 node_1_considers_var_1 + 5.8 node_1_considers_var_2
   + 2.6 node_1_considers_var_3 + 1.1 x144_is_in_node_15 <= 1.1
 R4821: - split_value_node_7 + 6.8 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.3 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x145_is_in_node_15 <= 1.1
 R4822: - split_value_node_3 + 6.8 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x145_is_in_node_15 <= 1.1
 R4823: - split_value_node_1 + 6.8 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x145_is_in_node_15 <= 1.1
 R4824: - split_value_node_7 + 6.4 node_7_considers_var_0
   + 2.6 node_7_considers_var_1 + 5.1 node_7_considers_var_2
   + 2 node_7_considers_var_3 + 1.1 x146_is_in_node_15 <= 1.1
 R4825: - split_value_node_3 + 6.4 node_3_considers_var_0
   + 2.6 node_3_considers_var_1 + 5.1 node_3_considers_var_2
   + 2 node_3_considers_var_3 + 1.1 x146_is_in_node_15 <= 1.1
 R4826: - split_value_node_1 + 6.4 node_1_considers_var_0
   + 2.6 node_1_considers_var_1 + 5.1 node_1_considers_var_2
   + 2 node_1_considers_var_3 + 1.1 x146_is_in_node_15 <= 1.1
 R4827: - split_value_node_7 + 6.6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.3 node_7_considers_var_2
   + 2.1 node_7_considers_var_3 + 1.1 x147_is_in_node_15 <= 1.1
 R4828: - split_value_node_3 + 6.6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.3 node_3_considers_var_2
   + 2.1 node_3_considers_var_3 + 1.1 x147_is_in_node_15 <= 1.1
 R4829: - split_value_node_1 + 6.6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.3 node_1_considers_var_2
   + 2.1 node_1_considers_var_3 + 1.1 x147_is_in_node_15 <= 1.1
 R4830: - split_value_node_7 + 6.3 node_7_considers_var_0
   + 3.5 node_7_considers_var_1 + 5.5 node_7_considers_var_2
   + 2.4 node_7_considers_var_3 + 1.1 x148_is_in_node_15 <= 1.1
 R4831: - split_value_node_3 + 6.3 node_3_considers_var_0
   + 3.5 node_3_considers_var_1 + 5.5 node_3_considers_var_2
   + 2.4 node_3_considers_var_3 + 1.1 x148_is_in_node_15 <= 1.1
 R4832: - split_value_node_1 + 6.3 node_1_considers_var_0
   + 3.5 node_1_considers_var_1 + 5.5 node_1_considers_var_2
   + 2.4 node_1_considers_var_3 + 1.1 x148_is_in_node_15 <= 1.1
 R4833: - split_value_node_7 + 6 node_7_considers_var_0
   + 3.1 node_7_considers_var_1 + 5.2 node_7_considers_var_2
   + 1.9 node_7_considers_var_3 + 1.1 x149_is_in_node_15 <= 1.1
 R4834: - split_value_node_3 + 6 node_3_considers_var_0
   + 3.1 node_3_considers_var_1 + 5.2 node_3_considers_var_2
   + 1.9 node_3_considers_var_3 + 1.1 x149_is_in_node_15 <= 1.1
 R4835: - split_value_node_1 + 6 node_1_considers_var_0
   + 3.1 node_1_considers_var_1 + 5.2 node_1_considers_var_2
   + 1.9 node_1_considers_var_3 + 1.1 x149_is_in_node_15 <= 1.1
Bounds
Binaries
 node_1_applies_split node_1_considers_var_0 node_1_considers_var_1
 node_1_considers_var_2 node_1_considers_var_3 node_2_applies_split
 node_2_considers_var_0 node_2_considers_var_1 node_2_considers_var_2
 node_2_considers_var_3 node_3_applies_split node_3_considers_var_0
 node_3_considers_var_1 node_3_considers_var_2 node_3_considers_var_3
 node_4_applies_split node_4_considers_var_0 node_4_considers_var_1
 node_4_considers_var_2 node_4_considers_var_3 node_5_applies_split
 node_5_considers_var_0 node_5_considers_var_1 node_5_considers_var_2
 node_5_considers_var_3 node_6_applies_split node_6_considers_var_0
 node_6_considers_var_1 node_6_considers_var_2 node_6_considers_var_3
 node_7_applies_split node_7_considers_var_0 node_7_considers_var_1
 node_7_considers_var_2 node_7_considers_var_3 node_8_contains_any_point
 x0_is_in_node_8 x1_is_in_node_8 x2_is_in_node_8 x3_is_in_node_8
 x4_is_in_node_8 x5_is_in_node_8 x6_is_in_node_8 x7_is_in_node_8
 x8_is_in_node_8 x9_is_in_node_8 x10_is_in_node_8 x11_is_in_node_8
 x12_is_in_node_8 x13_is_in_node_8 x14_is_in_node_8 x15_is_in_node_8
 x16_is_in_node_8 x17_is_in_node_8 x18_is_in_node_8 x19_is_in_node_8
 x20_is_in_node_8 x21_is_in_node_8 x22_is_in_node_8 x23_is_in_node_8
 x24_is_in_node_8 x25_is_in_node_8 x26_is_in_node_8 x27_is_in_node_8
 x28_is_in_node_8 x29_is_in_node_8 x30_is_in_node_8 x31_is_in_node_8
 x32_is_in_node_8 x33_is_in_node_8 x34_is_in_node_8 x35_is_in_node_8
 x36_is_in_node_8 x37_is_in_node_8 x38_is_in_node_8 x39_is_in_node_8
 x40_is_in_node_8 x41_is_in_node_8 x42_is_in_node_8 x43_is_in_node_8
 x44_is_in_node_8 x45_is_in_node_8 x46_is_in_node_8 x47_is_in_node_8
 x48_is_in_node_8 x49_is_in_node_8 x50_is_in_node_8 x51_is_in_node_8
 x52_is_in_node_8 x53_is_in_node_8 x54_is_in_node_8 x55_is_in_node_8
 x56_is_in_node_8 x57_is_in_node_8 x58_is_in_node_8 x59_is_in_node_8
 x60_is_in_node_8 x61_is_in_node_8 x62_is_in_node_8 x63_is_in_node_8
 x64_is_in_node_8 x65_is_in_node_8 x66_is_in_node_8 x67_is_in_node_8
 x68_is_in_node_8 x69_is_in_node_8 x70_is_in_node_8 x71_is_in_node_8
 x72_is_in_node_8 x73_is_in_node_8 x74_is_in_node_8 x75_is_in_node_8
 x76_is_in_node_8 x77_is_in_node_8 x78_is_in_node_8 x79_is_in_node_8
 x80_is_in_node_8 x81_is_in_node_8 x82_is_in_node_8 x83_is_in_node_8
 x84_is_in_node_8 x85_is_in_node_8 x86_is_in_node_8 x87_is_in_node_8
 x88_is_in_node_8 x89_is_in_node_8 x90_is_in_node_8 x91_is_in_node_8
 x92_is_in_node_8 x93_is_in_node_8 x94_is_in_node_8 x95_is_in_node_8
 x96_is_in_node_8 x97_is_in_node_8 x98_is_in_node_8 x99_is_in_node_8
 x100_is_in_node_8 x101_is_in_node_8 x102_is_in_node_8 x103_is_in_node_8
 x104_is_in_node_8 x105_is_in_node_8 x106_is_in_node_8 x107_is_in_node_8
 x108_is_in_node_8 x109_is_in_node_8 x110_is_in_node_8 x111_is_in_node_8
 x112_is_in_node_8 x113_is_in_node_8 x114_is_in_node_8 x115_is_in_node_8
 x116_is_in_node_8 x117_is_in_node_8 x118_is_in_node_8 x119_is_in_node_8
 x120_is_in_node_8 x121_is_in_node_8 x122_is_in_node_8 x123_is_in_node_8
 x124_is_in_node_8 x125_is_in_node_8 x126_is_in_node_8 x127_is_in_node_8
 x128_is_in_node_8 x129_is_in_node_8 x130_is_in_node_8 x131_is_in_node_8
 x132_is_in_node_8 x133_is_in_node_8 x134_is_in_node_8 x135_is_in_node_8
 x136_is_in_node_8 x137_is_in_node_8 x138_is_in_node_8 x139_is_in_node_8
 x140_is_in_node_8 x141_is_in_node_8 x142_is_in_node_8 x143_is_in_node_8
 x144_is_in_node_8 x145_is_in_node_8 x146_is_in_node_8 x147_is_in_node_8
 x148_is_in_node_8 x149_is_in_node_8 node_9_contains_any_point
 x0_is_in_node_9 x1_is_in_node_9 x2_is_in_node_9 x3_is_in_node_9
 x4_is_in_node_9 x5_is_in_node_9 x6_is_in_node_9 x7_is_in_node_9
 x8_is_in_node_9 x9_is_in_node_9 x10_is_in_node_9 x11_is_in_node_9
 x12_is_in_node_9 x13_is_in_node_9 x14_is_in_node_9 x15_is_in_node_9
 x16_is_in_node_9 x17_is_in_node_9 x18_is_in_node_9 x19_is_in_node_9
 x20_is_in_node_9 x21_is_in_node_9 x22_is_in_node_9 x23_is_in_node_9
 x24_is_in_node_9 x25_is_in_node_9 x26_is_in_node_9 x27_is_in_node_9
 x28_is_in_node_9 x29_is_in_node_9 x30_is_in_node_9 x31_is_in_node_9
 x32_is_in_node_9 x33_is_in_node_9 x34_is_in_node_9 x35_is_in_node_9
 x36_is_in_node_9 x37_is_in_node_9 x38_is_in_node_9 x39_is_in_node_9
 x40_is_in_node_9 x41_is_in_node_9 x42_is_in_node_9 x43_is_in_node_9
 x44_is_in_node_9 x45_is_in_node_9 x46_is_in_node_9 x47_is_in_node_9
 x48_is_in_node_9 x49_is_in_node_9 x50_is_in_node_9 x51_is_in_node_9
 x52_is_in_node_9 x53_is_in_node_9 x54_is_in_node_9 x55_is_in_node_9
 x56_is_in_node_9 x57_is_in_node_9 x58_is_in_node_9 x59_is_in_node_9
 x60_is_in_node_9 x61_is_in_node_9 x62_is_in_node_9 x63_is_in_node_9
 x64_is_in_node_9 x65_is_in_node_9 x66_is_in_node_9 x67_is_in_node_9
 x68_is_in_node_9 x69_is_in_node_9 x70_is_in_node_9 x71_is_in_node_9
 x72_is_in_node_9 x73_is_in_node_9 x74_is_in_node_9 x75_is_in_node_9
 x76_is_in_node_9 x77_is_in_node_9 x78_is_in_node_9 x79_is_in_node_9
 x80_is_in_node_9 x81_is_in_node_9 x82_is_in_node_9 x83_is_in_node_9
 x84_is_in_node_9 x85_is_in_node_9 x86_is_in_node_9 x87_is_in_node_9
 x88_is_in_node_9 x89_is_in_node_9 x90_is_in_node_9 x91_is_in_node_9
 x92_is_in_node_9 x93_is_in_node_9 x94_is_in_node_9 x95_is_in_node_9
 x96_is_in_node_9 x97_is_in_node_9 x98_is_in_node_9 x99_is_in_node_9
 x100_is_in_node_9 x101_is_in_node_9 x102_is_in_node_9 x103_is_in_node_9
 x104_is_in_node_9 x105_is_in_node_9 x106_is_in_node_9 x107_is_in_node_9
 x108_is_in_node_9 x109_is_in_node_9 x110_is_in_node_9 x111_is_in_node_9
 x112_is_in_node_9 x113_is_in_node_9 x114_is_in_node_9 x115_is_in_node_9
 x116_is_in_node_9 x117_is_in_node_9 x118_is_in_node_9 x119_is_in_node_9
 x120_is_in_node_9 x121_is_in_node_9 x122_is_in_node_9 x123_is_in_node_9
 x124_is_in_node_9 x125_is_in_node_9 x126_is_in_node_9 x127_is_in_node_9
 x128_is_in_node_9 x129_is_in_node_9 x130_is_in_node_9 x131_is_in_node_9
 x132_is_in_node_9 x133_is_in_node_9 x134_is_in_node_9 x135_is_in_node_9
 x136_is_in_node_9 x137_is_in_node_9 x138_is_in_node_9 x139_is_in_node_9
 x140_is_in_node_9 x141_is_in_node_9 x142_is_in_node_9 x143_is_in_node_9
 x144_is_in_node_9 x145_is_in_node_9 x146_is_in_node_9 x147_is_in_node_9
 x148_is_in_node_9 x149_is_in_node_9 node_10_contains_any_point
 x0_is_in_node_10 x1_is_in_node_10 x2_is_in_node_10 x3_is_in_node_10
 x4_is_in_node_10 x5_is_in_node_10 x6_is_in_node_10 x7_is_in_node_10
 x8_is_in_node_10 x9_is_in_node_10 x10_is_in_node_10 x11_is_in_node_10
 x12_is_in_node_10 x13_is_in_node_10 x14_is_in_node_10 x15_is_in_node_10
 x16_is_in_node_10 x17_is_in_node_10 x18_is_in_node_10 x19_is_in_node_10
 x20_is_in_node_10 x21_is_in_node_10 x22_is_in_node_10 x23_is_in_node_10
 x24_is_in_node_10 x25_is_in_node_10 x26_is_in_node_10 x27_is_in_node_10
 x28_is_in_node_10 x29_is_in_node_10 x30_is_in_node_10 x31_is_in_node_10
 x32_is_in_node_10 x33_is_in_node_10 x34_is_in_node_10 x35_is_in_node_10
 x36_is_in_node_10 x37_is_in_node_10 x38_is_in_node_10 x39_is_in_node_10
 x40_is_in_node_10 x41_is_in_node_10 x42_is_in_node_10 x43_is_in_node_10
 x44_is_in_node_10 x45_is_in_node_10 x46_is_in_node_10 x47_is_in_node_10
 x48_is_in_node_10 x49_is_in_node_10 x50_is_in_node_10 x51_is_in_node_10
 x52_is_in_node_10 x53_is_in_node_10 x54_is_in_node_10 x55_is_in_node_10
 x56_is_in_node_10 x57_is_in_node_10 x58_is_in_node_10 x59_is_in_node_10
 x60_is_in_node_10 x61_is_in_node_10 x62_is_in_node_10 x63_is_in_node_10
 x64_is_in_node_10 x65_is_in_node_10 x66_is_in_node_10 x67_is_in_node_10
 x68_is_in_node_10 x69_is_in_node_10 x70_is_in_node_10 x71_is_in_node_10
 x72_is_in_node_10 x73_is_in_node_10 x74_is_in_node_10 x75_is_in_node_10
 x76_is_in_node_10 x77_is_in_node_10 x78_is_in_node_10 x79_is_in_node_10
 x80_is_in_node_10 x81_is_in_node_10 x82_is_in_node_10 x83_is_in_node_10
 x84_is_in_node_10 x85_is_in_node_10 x86_is_in_node_10 x87_is_in_node_10
 x88_is_in_node_10 x89_is_in_node_10 x90_is_in_node_10 x91_is_in_node_10
 x92_is_in_node_10 x93_is_in_node_10 x94_is_in_node_10 x95_is_in_node_10
 x96_is_in_node_10 x97_is_in_node_10 x98_is_in_node_10 x99_is_in_node_10
 x100_is_in_node_10 x101_is_in_node_10 x102_is_in_node_10
 x103_is_in_node_10 x104_is_in_node_10 x105_is_in_node_10
 x106_is_in_node_10 x107_is_in_node_10 x108_is_in_node_10
 x109_is_in_node_10 x110_is_in_node_10 x111_is_in_node_10
 x112_is_in_node_10 x113_is_in_node_10 x114_is_in_node_10
 x115_is_in_node_10 x116_is_in_node_10 x117_is_in_node_10
 x118_is_in_node_10 x119_is_in_node_10 x120_is_in_node_10
 x121_is_in_node_10 x122_is_in_node_10 x123_is_in_node_10
 x124_is_in_node_10 x125_is_in_node_10 x126_is_in_node_10
 x127_is_in_node_10 x128_is_in_node_10 x129_is_in_node_10
 x130_is_in_node_10 x131_is_in_node_10 x132_is_in_node_10
 x133_is_in_node_10 x134_is_in_node_10 x135_is_in_node_10
 x136_is_in_node_10 x137_is_in_node_10 x138_is_in_node_10
 x139_is_in_node_10 x140_is_in_node_10 x141_is_in_node_10
 x142_is_in_node_10 x143_is_in_node_10 x144_is_in_node_10
 x145_is_in_node_10 x146_is_in_node_10 x147_is_in_node_10
 x148_is_in_node_10 x149_is_in_node_10 node_11_contains_any_point
 x0_is_in_node_11 x1_is_in_node_11 x2_is_in_node_11 x3_is_in_node_11
 x4_is_in_node_11 x5_is_in_node_11 x6_is_in_node_11 x7_is_in_node_11
 x8_is_in_node_11 x9_is_in_node_11 x10_is_in_node_11 x11_is_in_node_11
 x12_is_in_node_11 x13_is_in_node_11 x14_is_in_node_11 x15_is_in_node_11
 x16_is_in_node_11 x17_is_in_node_11 x18_is_in_node_11 x19_is_in_node_11
 x20_is_in_node_11 x21_is_in_node_11 x22_is_in_node_11 x23_is_in_node_11
 x24_is_in_node_11 x25_is_in_node_11 x26_is_in_node_11 x27_is_in_node_11
 x28_is_in_node_11 x29_is_in_node_11 x30_is_in_node_11 x31_is_in_node_11
 x32_is_in_node_11 x33_is_in_node_11 x34_is_in_node_11 x35_is_in_node_11
 x36_is_in_node_11 x37_is_in_node_11 x38_is_in_node_11 x39_is_in_node_11
 x40_is_in_node_11 x41_is_in_node_11 x42_is_in_node_11 x43_is_in_node_11
 x44_is_in_node_11 x45_is_in_node_11 x46_is_in_node_11 x47_is_in_node_11
 x48_is_in_node_11 x49_is_in_node_11 x50_is_in_node_11 x51_is_in_node_11
 x52_is_in_node_11 x53_is_in_node_11 x54_is_in_node_11 x55_is_in_node_11
 x56_is_in_node_11 x57_is_in_node_11 x58_is_in_node_11 x59_is_in_node_11
 x60_is_in_node_11 x61_is_in_node_11 x62_is_in_node_11 x63_is_in_node_11
 x64_is_in_node_11 x65_is_in_node_11 x66_is_in_node_11 x67_is_in_node_11
 x68_is_in_node_11 x69_is_in_node_11 x70_is_in_node_11 x71_is_in_node_11
 x72_is_in_node_11 x73_is_in_node_11 x74_is_in_node_11 x75_is_in_node_11
 x76_is_in_node_11 x77_is_in_node_11 x78_is_in_node_11 x79_is_in_node_11
 x80_is_in_node_11 x81_is_in_node_11 x82_is_in_node_11 x83_is_in_node_11
 x84_is_in_node_11 x85_is_in_node_11 x86_is_in_node_11 x87_is_in_node_11
 x88_is_in_node_11 x89_is_in_node_11 x90_is_in_node_11 x91_is_in_node_11
 x92_is_in_node_11 x93_is_in_node_11 x94_is_in_node_11 x95_is_in_node_11
 x96_is_in_node_11 x97_is_in_node_11 x98_is_in_node_11 x99_is_in_node_11
 x100_is_in_node_11 x101_is_in_node_11 x102_is_in_node_11
 x103_is_in_node_11 x104_is_in_node_11 x105_is_in_node_11
 x106_is_in_node_11 x107_is_in_node_11 x108_is_in_node_11
 x109_is_in_node_11 x110_is_in_node_11 x111_is_in_node_11
 x112_is_in_node_11 x113_is_in_node_11 x114_is_in_node_11
 x115_is_in_node_11 x116_is_in_node_11 x117_is_in_node_11
 x118_is_in_node_11 x119_is_in_node_11 x120_is_in_node_11
 x121_is_in_node_11 x122_is_in_node_11 x123_is_in_node_11
 x124_is_in_node_11 x125_is_in_node_11 x126_is_in_node_11
 x127_is_in_node_11 x128_is_in_node_11 x129_is_in_node_11
 x130_is_in_node_11 x131_is_in_node_11 x132_is_in_node_11
 x133_is_in_node_11 x134_is_in_node_11 x135_is_in_node_11
 x136_is_in_node_11 x137_is_in_node_11 x138_is_in_node_11
 x139_is_in_node_11 x140_is_in_node_11 x141_is_in_node_11
 x142_is_in_node_11 x143_is_in_node_11 x144_is_in_node_11
 x145_is_in_node_11 x146_is_in_node_11 x147_is_in_node_11
 x148_is_in_node_11 x149_is_in_node_11 node_12_contains_any_point
 x0_is_in_node_12 x1_is_in_node_12 x2_is_in_node_12 x3_is_in_node_12
 x4_is_in_node_12 x5_is_in_node_12 x6_is_in_node_12 x7_is_in_node_12
 x8_is_in_node_12 x9_is_in_node_12 x10_is_in_node_12 x11_is_in_node_12
 x12_is_in_node_12 x13_is_in_node_12 x14_is_in_node_12 x15_is_in_node_12
 x16_is_in_node_12 x17_is_in_node_12 x18_is_in_node_12 x19_is_in_node_12
 x20_is_in_node_12 x21_is_in_node_12 x22_is_in_node_12 x23_is_in_node_12
 x24_is_in_node_12 x25_is_in_node_12 x26_is_in_node_12 x27_is_in_node_12
 x28_is_in_node_12 x29_is_in_node_12 x30_is_in_node_12 x31_is_in_node_12
 x32_is_in_node_12 x33_is_in_node_12 x34_is_in_node_12 x35_is_in_node_12
 x36_is_in_node_12 x37_is_in_node_12 x38_is_in_node_12 x39_is_in_node_12
 x40_is_in_node_12 x41_is_in_node_12 x42_is_in_node_12 x43_is_in_node_12
 x44_is_in_node_12 x45_is_in_node_12 x46_is_in_node_12 x47_is_in_node_12
 x48_is_in_node_12 x49_is_in_node_12 x50_is_in_node_12 x51_is_in_node_12
 x52_is_in_node_12 x53_is_in_node_12 x54_is_in_node_12 x55_is_in_node_12
 x56_is_in_node_12 x57_is_in_node_12 x58_is_in_node_12 x59_is_in_node_12
 x60_is_in_node_12 x61_is_in_node_12 x62_is_in_node_12 x63_is_in_node_12
 x64_is_in_node_12 x65_is_in_node_12 x66_is_in_node_12 x67_is_in_node_12
 x68_is_in_node_12 x69_is_in_node_12 x70_is_in_node_12 x71_is_in_node_12
 x72_is_in_node_12 x73_is_in_node_12 x74_is_in_node_12 x75_is_in_node_12
 x76_is_in_node_12 x77_is_in_node_12 x78_is_in_node_12 x79_is_in_node_12
 x80_is_in_node_12 x81_is_in_node_12 x82_is_in_node_12 x83_is_in_node_12
 x84_is_in_node_12 x85_is_in_node_12 x86_is_in_node_12 x87_is_in_node_12
 x88_is_in_node_12 x89_is_in_node_12 x90_is_in_node_12 x91_is_in_node_12
 x92_is_in_node_12 x93_is_in_node_12 x94_is_in_node_12 x95_is_in_node_12
 x96_is_in_node_12 x97_is_in_node_12 x98_is_in_node_12 x99_is_in_node_12
 x100_is_in_node_12 x101_is_in_node_12 x102_is_in_node_12
 x103_is_in_node_12 x104_is_in_node_12 x105_is_in_node_12
 x106_is_in_node_12 x107_is_in_node_12 x108_is_in_node_12
 x109_is_in_node_12 x110_is_in_node_12 x111_is_in_node_12
 x112_is_in_node_12 x113_is_in_node_12 x114_is_in_node_12
 x115_is_in_node_12 x116_is_in_node_12 x117_is_in_node_12
 x118_is_in_node_12 x119_is_in_node_12 x120_is_in_node_12
 x121_is_in_node_12 x122_is_in_node_12 x123_is_in_node_12
 x124_is_in_node_12 x125_is_in_node_12 x126_is_in_node_12
 x127_is_in_node_12 x128_is_in_node_12 x129_is_in_node_12
 x130_is_in_node_12 x131_is_in_node_12 x132_is_in_node_12
 x133_is_in_node_12 x134_is_in_node_12 x135_is_in_node_12
 x136_is_in_node_12 x137_is_in_node_12 x138_is_in_node_12
 x139_is_in_node_12 x140_is_in_node_12 x141_is_in_node_12
 x142_is_in_node_12 x143_is_in_node_12 x144_is_in_node_12
 x145_is_in_node_12 x146_is_in_node_12 x147_is_in_node_12
 x148_is_in_node_12 x149_is_in_node_12 node_13_contains_any_point
 x0_is_in_node_13 x1_is_in_node_13 x2_is_in_node_13 x3_is_in_node_13
 x4_is_in_node_13 x5_is_in_node_13 x6_is_in_node_13 x7_is_in_node_13
 x8_is_in_node_13 x9_is_in_node_13 x10_is_in_node_13 x11_is_in_node_13
 x12_is_in_node_13 x13_is_in_node_13 x14_is_in_node_13 x15_is_in_node_13
 x16_is_in_node_13 x17_is_in_node_13 x18_is_in_node_13 x19_is_in_node_13
 x20_is_in_node_13 x21_is_in_node_13 x22_is_in_node_13 x23_is_in_node_13
 x24_is_in_node_13 x25_is_in_node_13 x26_is_in_node_13 x27_is_in_node_13
 x28_is_in_node_13 x29_is_in_node_13 x30_is_in_node_13 x31_is_in_node_13
 x32_is_in_node_13 x33_is_in_node_13 x34_is_in_node_13 x35_is_in_node_13
 x36_is_in_node_13 x37_is_in_node_13 x38_is_in_node_13 x39_is_in_node_13
 x40_is_in_node_13 x41_is_in_node_13 x42_is_in_node_13 x43_is_in_node_13
 x44_is_in_node_13 x45_is_in_node_13 x46_is_in_node_13 x47_is_in_node_13
 x48_is_in_node_13 x49_is_in_node_13 x50_is_in_node_13 x51_is_in_node_13
 x52_is_in_node_13 x53_is_in_node_13 x54_is_in_node_13 x55_is_in_node_13
 x56_is_in_node_13 x57_is_in_node_13 x58_is_in_node_13 x59_is_in_node_13
 x60_is_in_node_13 x61_is_in_node_13 x62_is_in_node_13 x63_is_in_node_13
 x64_is_in_node_13 x65_is_in_node_13 x66_is_in_node_13 x67_is_in_node_13
 x68_is_in_node_13 x69_is_in_node_13 x70_is_in_node_13 x71_is_in_node_13
 x72_is_in_node_13 x73_is_in_node_13 x74_is_in_node_13 x75_is_in_node_13
 x76_is_in_node_13 x77_is_in_node_13 x78_is_in_node_13 x79_is_in_node_13
 x80_is_in_node_13 x81_is_in_node_13 x82_is_in_node_13 x83_is_in_node_13
 x84_is_in_node_13 x85_is_in_node_13 x86_is_in_node_13 x87_is_in_node_13
 x88_is_in_node_13 x89_is_in_node_13 x90_is_in_node_13 x91_is_in_node_13
 x92_is_in_node_13 x93_is_in_node_13 x94_is_in_node_13 x95_is_in_node_13
 x96_is_in_node_13 x97_is_in_node_13 x98_is_in_node_13 x99_is_in_node_13
 x100_is_in_node_13 x101_is_in_node_13 x102_is_in_node_13
 x103_is_in_node_13 x104_is_in_node_13 x105_is_in_node_13
 x106_is_in_node_13 x107_is_in_node_13 x108_is_in_node_13
 x109_is_in_node_13 x110_is_in_node_13 x111_is_in_node_13
 x112_is_in_node_13 x113_is_in_node_13 x114_is_in_node_13
 x115_is_in_node_13 x116_is_in_node_13 x117_is_in_node_13
 x118_is_in_node_13 x119_is_in_node_13 x120_is_in_node_13
 x121_is_in_node_13 x122_is_in_node_13 x123_is_in_node_13
 x124_is_in_node_13 x125_is_in_node_13 x126_is_in_node_13
 x127_is_in_node_13 x128_is_in_node_13 x129_is_in_node_13
 x130_is_in_node_13 x131_is_in_node_13 x132_is_in_node_13
 x133_is_in_node_13 x134_is_in_node_13 x135_is_in_node_13
 x136_is_in_node_13 x137_is_in_node_13 x138_is_in_node_13
 x139_is_in_node_13 x140_is_in_node_13 x141_is_in_node_13
 x142_is_in_node_13 x143_is_in_node_13 x144_is_in_node_13
 x145_is_in_node_13 x146_is_in_node_13 x147_is_in_node_13
 x148_is_in_node_13 x149_is_in_node_13 node_14_contains_any_point
 x0_is_in_node_14 x1_is_in_node_14 x2_is_in_node_14 x3_is_in_node_14
 x4_is_in_node_14 x5_is_in_node_14 x6_is_in_node_14 x7_is_in_node_14
 x8_is_in_node_14 x9_is_in_node_14 x10_is_in_node_14 x11_is_in_node_14
 x12_is_in_node_14 x13_is_in_node_14 x14_is_in_node_14 x15_is_in_node_14
 x16_is_in_node_14 x17_is_in_node_14 x18_is_in_node_14 x19_is_in_node_14
 x20_is_in_node_14 x21_is_in_node_14 x22_is_in_node_14 x23_is_in_node_14
 x24_is_in_node_14 x25_is_in_node_14 x26_is_in_node_14 x27_is_in_node_14
 x28_is_in_node_14 x29_is_in_node_14 x30_is_in_node_14 x31_is_in_node_14
 x32_is_in_node_14 x33_is_in_node_14 x34_is_in_node_14 x35_is_in_node_14
 x36_is_in_node_14 x37_is_in_node_14 x38_is_in_node_14 x39_is_in_node_14
 x40_is_in_node_14 x41_is_in_node_14 x42_is_in_node_14 x43_is_in_node_14
 x44_is_in_node_14 x45_is_in_node_14 x46_is_in_node_14 x47_is_in_node_14
 x48_is_in_node_14 x49_is_in_node_14 x50_is_in_node_14 x51_is_in_node_14
 x52_is_in_node_14 x53_is_in_node_14 x54_is_in_node_14 x55_is_in_node_14
 x56_is_in_node_14 x57_is_in_node_14 x58_is_in_node_14 x59_is_in_node_14
 x60_is_in_node_14 x61_is_in_node_14 x62_is_in_node_14 x63_is_in_node_14
 x64_is_in_node_14 x65_is_in_node_14 x66_is_in_node_14 x67_is_in_node_14
 x68_is_in_node_14 x69_is_in_node_14 x70_is_in_node_14 x71_is_in_node_14
 x72_is_in_node_14 x73_is_in_node_14 x74_is_in_node_14 x75_is_in_node_14
 x76_is_in_node_14 x77_is_in_node_14 x78_is_in_node_14 x79_is_in_node_14
 x80_is_in_node_14 x81_is_in_node_14 x82_is_in_node_14 x83_is_in_node_14
 x84_is_in_node_14 x85_is_in_node_14 x86_is_in_node_14 x87_is_in_node_14
 x88_is_in_node_14 x89_is_in_node_14 x90_is_in_node_14 x91_is_in_node_14
 x92_is_in_node_14 x93_is_in_node_14 x94_is_in_node_14 x95_is_in_node_14
 x96_is_in_node_14 x97_is_in_node_14 x98_is_in_node_14 x99_is_in_node_14
 x100_is_in_node_14 x101_is_in_node_14 x102_is_in_node_14
 x103_is_in_node_14 x104_is_in_node_14 x105_is_in_node_14
 x106_is_in_node_14 x107_is_in_node_14 x108_is_in_node_14
 x109_is_in_node_14 x110_is_in_node_14 x111_is_in_node_14
 x112_is_in_node_14 x113_is_in_node_14 x114_is_in_node_14
 x115_is_in_node_14 x116_is_in_node_14 x117_is_in_node_14
 x118_is_in_node_14 x119_is_in_node_14 x120_is_in_node_14
 x121_is_in_node_14 x122_is_in_node_14 x123_is_in_node_14
 x124_is_in_node_14 x125_is_in_node_14 x126_is_in_node_14
 x127_is_in_node_14 x128_is_in_node_14 x129_is_in_node_14
 x130_is_in_node_14 x131_is_in_node_14 x132_is_in_node_14
 x133_is_in_node_14 x134_is_in_node_14 x135_is_in_node_14
 x136_is_in_node_14 x137_is_in_node_14 x138_is_in_node_14
 x139_is_in_node_14 x140_is_in_node_14 x141_is_in_node_14
 x142_is_in_node_14 x143_is_in_node_14 x144_is_in_node_14
 x145_is_in_node_14 x146_is_in_node_14 x147_is_in_node_14
 x148_is_in_node_14 x149_is_in_node_14 node_15_contains_any_point
 x0_is_in_node_15 x1_is_in_node_15 x2_is_in_node_15 x3_is_in_node_15
 x4_is_in_node_15 x5_is_in_node_15 x6_is_in_node_15 x7_is_in_node_15
 x8_is_in_node_15 x9_is_in_node_15 x10_is_in_node_15 x11_is_in_node_15
 x12_is_in_node_15 x13_is_in_node_15 x14_is_in_node_15 x15_is_in_node_15
 x16_is_in_node_15 x17_is_in_node_15 x18_is_in_node_15 x19_is_in_node_15
 x20_is_in_node_15 x21_is_in_node_15 x22_is_in_node_15 x23_is_in_node_15
 x24_is_in_node_15 x25_is_in_node_15 x26_is_in_node_15 x27_is_in_node_15
 x28_is_in_node_15 x29_is_in_node_15 x30_is_in_node_15 x31_is_in_node_15
 x32_is_in_node_15 x33_is_in_node_15 x34_is_in_node_15 x35_is_in_node_15
 x36_is_in_node_15 x37_is_in_node_15 x38_is_in_node_15 x39_is_in_node_15
 x40_is_in_node_15 x41_is_in_node_15 x42_is_in_node_15 x43_is_in_node_15
 x44_is_in_node_15 x45_is_in_node_15 x46_is_in_node_15 x47_is_in_node_15
 x48_is_in_node_15 x49_is_in_node_15 x50_is_in_node_15 x51_is_in_node_15
 x52_is_in_node_15 x53_is_in_node_15 x54_is_in_node_15 x55_is_in_node_15
 x56_is_in_node_15 x57_is_in_node_15 x58_is_in_node_15 x59_is_in_node_15
 x60_is_in_node_15 x61_is_in_node_15 x62_is_in_node_15 x63_is_in_node_15
 x64_is_in_node_15 x65_is_in_node_15 x66_is_in_node_15 x67_is_in_node_15
 x68_is_in_node_15 x69_is_in_node_15 x70_is_in_node_15 x71_is_in_node_15
 x72_is_in_node_15 x73_is_in_node_15 x74_is_in_node_15 x75_is_in_node_15
 x76_is_in_node_15 x77_is_in_node_15 x78_is_in_node_15 x79_is_in_node_15
 x80_is_in_node_15 x81_is_in_node_15 x82_is_in_node_15 x83_is_in_node_15
 x84_is_in_node_15 x85_is_in_node_15 x86_is_in_node_15 x87_is_in_node_15
 x88_is_in_node_15 x89_is_in_node_15 x90_is_in_node_15 x91_is_in_node_15
 x92_is_in_node_15 x93_is_in_node_15 x94_is_in_node_15 x95_is_in_node_15
 x96_is_in_node_15 x97_is_in_node_15 x98_is_in_node_15 x99_is_in_node_15
 x100_is_in_node_15 x101_is_in_node_15 x102_is_in_node_15
 x103_is_in_node_15 x104_is_in_node_15 x105_is_in_node_15
 x106_is_in_node_15 x107_is_in_node_15 x108_is_in_node_15
 x109_is_in_node_15 x110_is_in_node_15 x111_is_in_node_15
 x112_is_in_node_15 x113_is_in_node_15 x114_is_in_node_15
 x115_is_in_node_15 x116_is_in_node_15 x117_is_in_node_15
 x118_is_in_node_15 x119_is_in_node_15 x120_is_in_node_15
 x121_is_in_node_15 x122_is_in_node_15 x123_is_in_node_15
 x124_is_in_node_15 x125_is_in_node_15 x126_is_in_node_15
 x127_is_in_node_15 x128_is_in_node_15 x129_is_in_node_15
 x130_is_in_node_15 x131_is_in_node_15 x132_is_in_node_15
 x133_is_in_node_15 x134_is_in_node_15 x135_is_in_node_15
 x136_is_in_node_15 x137_is_in_node_15 x138_is_in_node_15
 x139_is_in_node_15 x140_is_in_node_15 x141_is_in_node_15
 x142_is_in_node_15 x143_is_in_node_15 x144_is_in_node_15
 x145_is_in_node_15 x146_is_in_node_15 x147_is_in_node_15
 x148_is_in_node_15 x149_is_in_node_15
End
