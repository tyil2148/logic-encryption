import random

def replace(netlist, selected_gates, t, output, inputs, outputs, percent): # This function creates the new encrypted circuit
    newfile = "%s" % output
    if percent == 0:
        randomoutputs = []
    else:
        randomoutputs = random.sample(list(outputs), (len(outputs) // (100 // percent)))
    with open(newfile, "w") as file:
        if t == "gshe":
            import gshe_hybrid as g
        elif t == "dwm":
            import dwm_hybrid as g
        elif t == "tvd":
            import tvd_hybrid as g
        elif t == "sinw":
            import sinw_hybrid as g
        elif t == "dummy":
            import dummy_hybrid as g
        elif t == "nona":
            import nona_hybrid as g
        with open(netlist) as f:
            i = 1
            k = 0
            for line in f:
                s = line.split()
                if len(s) > 0:
                    if line.startswith("# c"):
                        file.write("%s" % line)
                        if t == "gshe":
                            for num in range(1, 415):
                                file.write(f"INPUT(keyinput{num})\n")
                        elif t == "dwm" or t == "tvd":
                            for num in range (1, 190):
                                file.write(f"INPUT(keyinput{num})\n")
                        elif t == "sinw" or t == "dummy":
                            for num in range (1, 145):
                                file.write(f"INPUT(keyinput{num})\n")
                        elif t == "nona":
                            for num in range(1, 100):
                                file.write(f"INPUT(keyinput{num})\n")
                    if s[0] in selected_gates:
                        for l in g.make(line, i, inputs):
                            file.write("%s\n" % l)
                        if i == 45:
                            if t == "gshe":
                                x = 406
                            elif t == "dwm" or t == "tvd":
                                x = 181
                            elif t == "sinw" or t == "dummy":
                                x = 136
                            elif t == "nona":
                                x = 91
                            # write big SAT-Resilient Block output
                            file.write("reinstatein = AND(srb1, srb2, srb3, srb4, srb5, srb6, srb7, srb8, srb9, srb10, srb11, srb12, srb13, srb14, srb15, srb16, srb17, srb18, srb19, srb20, srb21, srb22, srb23, srb24, srb25, srb26, srb27, srb28, srb29, srb30, srb31, srb32, srb33, srb34, srb35, srb36, srb37, srb38, srb39, srb40, srb41, srb42, srb43, srb44, srb45)\n")
                            
                            # write reinstate block based on GSHE -- A = reinstatein, B = keyinputX

                            htriggera = random.choice(list(inputs))
                            htriggerb = random.choice(list(inputs))
                            zeroone = random.choice(list(inputs))

                            keys = ["keyinput%d" % (x + 1), "keyinput%d" % (x + 2), "keyinput%d" % (x + 3), "keyinput%d" % (x + 4), "keyinput%d" % (x + 5), "keyinput%d" % (x + 6), "keyinput%d" % (x + 7), "keyinput%d" % (x + 8)]

                            file.write("zero_f = XOR(%s, %s)\n" % (zeroone, zeroone))
                            file.write("one_f = XNOR(%s, %s)\n" % (zeroone, zeroone))

                            rng = ["zero_f, one_f"]

                            file.write("htriggerA_f = BUF(%s)\n" % htriggera)
                            file.write("htriggerB_f = BUF(%s)\n" % htriggerb)

                            file.write("indA_f = AND(reinstatein, %s, %s)\n" % (htriggera, random.choice(rng)))

                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 1, x + 1))

                            file.write("w1A_f = AND(indA_f, not_keyinput%d)\n" % (x + 1))
                            file.write("w2A_f = AND(reinstatein, keyinput%d)\n" % (x + 1))

                            file.write("dynamicA_f = OR(w1A_f, w2A_f)\n")
                            file.write("not_dynamicA_f = NOT(dynamicA_f)\n")

                            file.write("indB_f = AND(keyinput%d, %s, %s)\n" % (x, htriggerb, random.choice(rng)))

                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 2, x + 2))

                            file.write("w1B_f = AND(indB_f, not_keyinput%d)\n" % (x + 2))
                            file.write("w2B_f = AND(keyinput%d, keyinput%d)\n" % (x, x + 2))

                            file.write("dynamicB_f = OR(w1B_f, w2B_f)\n")
                            file.write("not_dynamicB_f = NOT(dynamicB_f)\n")

                            file.write("i0_f = NAND(dynamicA_f, dynamicB_f)\n") # a NAND b
                            file.write("i1_f = AND(dynamicA_f, dynamicB_f)\n") # a AND b
                            file.write("i2_f = NOR(dynamicA_f, dynamicB_f)\n") # a NOR b
                            file.write("i3_f = OR(dynamicA_f, dynamicB_f)\n") # a OR b
                            file.write("i4_f = XOR(dynamicA_f, dynamicB_f)\n") # a XOR b
                            file.write("i5_f = XNOR(dynamicA_f, dynamicB_f)\n") # a XNOR b
                            file.write("i6_f = NOT(dynamicA_f)\n") # NOT a
                            file.write("i7_f = BUF(dynamicA_f)\n") # BUF a
                            file.write("i8_f = AND(dynamicA_f, not_dynamicB_f)\n") # a AND NOT b
                            file.write("i9_f = AND(not_dynamicA_f, dynamicB_f)\n") # NOT a AND b
                            file.write("i10_f = OR(dynamicA_f, not_dynamicB_f)\n") # a OR NOT b
                            file.write("i11_f = OR(not_dynamicA_f, dynamicB_f)\n") # NOT a OR b
                            file.write("i12_f = NOT(dynamicB_f)\n") # NOT b
                            file.write("i13_f = BUF(dynamicB_f)\n") # BUF b
                            file.write("i14_f = BUF(one_f)\n") # TRUE
                            file.write("i15_f = BUF(zero_f)\n") # FALSE

                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 3, x + 3))
                            file.write("w0s0_f = AND(zero_f, not_keyinput%d)\n" % (x + 3))
                            file.write("w1s0_f = AND(one_f, keyinput%d)\n" % (x + 3))
                            file.write("s1_f = OR(w0s0_f, w1s0_f)\n")
                            file.write("not_s1_f = NOT(s1_f)\n")

                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 4, x + 4))
                            file.write("w2s0_f = AND(s1_f, not_keyinput%d)\n" % (x + 4))
                            file.write("w3s0_f = AND(not_s1_f, keyinput%d)\n" % (x + 4))
                            file.write("s0_f = OR(w2s0_f, w3s0_f)\n")
                            file.write("not_s0_f = NOT(s0_f)\n")

                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 5, x + 5))
                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 6, x + 6))
                            file.write("w0s2_f = AND(zero_f, not_keyinput%d, not_keyinput%d)\n" % (x + 5, x + 6))
                            file.write("w1s2_f = AND(one_f, not_keyinput%d, keyinput%d)\n" % (x + 5, x + 6))
                            file.write("w2s2_f = AND(keyinput%d, keyinput%d, not_keyinput%d)\n" % (x, x + 5, x + 6))
                            file.write("w3s2_f = AND(reinstatein, keyinput%d, keyinput%d)\n" % (x + 5, x + 6))
                            file.write("s2_f = OR(w0s2_f, w1s2_f, w2s2_f, w3s2_f)\n")
                            file.write("not_s2_f = NOT(s2_f)\n")

                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 7, x + 7))
                            file.write("not_keyinput%d = NOT(keyinput%d)\n" % (x + 8, x + 8))
                            file.write("w0s3_f = AND(zero_f, not_keyinput%d, not_keyinput%d)\n" % (x + 7, x + 8))
                            file.write("w1s3_f = AND(one_f, not_keyinput%d, keyinput%d)\n" % (x + 7, x + 8))
                            file.write("w2s3_f = AND(keyinput%d, keyinput%d, not_keyinput%d)\n" % (x, x + 7, x + 8))
                            file.write("w3s3_f = AND(reinstatein, keyinput%d, keyinput%d)\n" % (x + 7, x + 8))
                            file.write("s3_f = OR(w0s3_f, w1s3_f, w2s3_f, w3s3_f)\n")
                            file.write("not_s3_f = NOT(s3_f)\n")

                            file.write("and0mux0_f = AND(i0_f, not_s0_f, not_s1_f)\n")
                            file.write("and1mux0_f = AND(i1_f, not_s0_f, s1_f)\n")
                            file.write("and2mux0_f = AND(i2_f, s0_f, not_s1_f)\n")
                            file.write("and3mux0_f = AND(i3_f, s0_f, s1_f)\n")
                            file.write("mux0_f = OR(and0mux0_f, and1mux0_f, and2mux0_f, and3mux0_f)\n")

                            file.write("and0mux1_f = AND(i4_f, not_s0_f, not_s1_f)\n")
                            file.write("and1mux1_f = AND(i5_f, not_s0_f, s1_f)\n")
                            file.write("and2mux1_f = AND(i6_f, s0_f, not_s1_f)\n")
                            file.write("and3mux1_f = AND(i7_f, s0_f, s1_f)\n")
                            file.write("mux1_f = OR(and0mux1_f, and1mux1_f, and2mux1_f, and3mux1_f)\n")

                            file.write("and0mux2_f = AND(i8_f, not_s0_f, not_s1_f)\n")
                            file.write("and1mux2_f = AND(i9_f, not_s0_f, s1_f)\n")
                            file.write("and2mux2_f = AND(i10_f, s0_f, not_s1_f)\n")
                            file.write("and3mux2_f = AND(i11_f, s0_f, s1_f)\n")
                            file.write("mux2_f = OR(and0mux2_f, and1mux2_f, and2mux2_f, and3mux2_f)\n")

                            file.write("and0mux3_f = AND(i12_f, not_s0_f, not_s1_f)\n")
                            file.write("and1mux3_f = AND(i13_f, not_s0_f, s1_f)\n")
                            file.write("and2mux3_f = AND(i14_f, s0_f, not_s1_f)\n")
                            file.write("and3mux3_f = AND(i15_f, s0_f, s1_f)\n")
                            file.write("mux3_f = OR(and0mux3_f, and1mux3_f, and2mux3_f, and3mux3_f)\n")

                            file.write("and0mux4_f = AND(mux0_f, not_s2_f, not_s3_f)\n")
                            file.write("and1mux4_f = AND(mux1_f, not_s2_f, s3_f)\n")
                            file.write("and2mux4_f = AND(mux2_f, s2_f, not_s3_f)\n")
                            file.write("and3mux4_f = AND(mux3_f, s2_f, s3_f)\n")
                            file.write("reinstateout = OR(and0mux4_f, and1mux4_f, and2mux4_f, and3mux4_f)\n")
                        i += 1
                    else:
                        if s[0] in randomoutputs:
                            n = randomoutputs.index(s[0])
                            line = line[len(s[0]):]
                            file.write("newout%d%s\n" % (n, line))
                            file.write("newin%d = XOR(newout%d, reinstatein)\n" % (n, n))
                            file.write("%s = XOR(newin%d, reinstateout)\n" % (s[0], n))
                        else:
                            file.write("%s" % line)

    print("File Done!")

