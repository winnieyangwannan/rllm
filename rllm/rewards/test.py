

# inout = { "inputs": [ "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n" ], "outputs": [ "1\n3 \n-1\n0\n\n2\n1 2 \n" ] }

# inout = {"inputs": ["3\n4 5\n6 3\n10 2\n"], "outputs": ["5\n3 4\n4 4 1 2\n"]}
# inputs = inout["inputs"]

# # print(inputs)

# def process_input_output(inputs, outputs):
#     # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
#     try:
#         if isinstance(inputs[0], dict):
#             inputs = [{int(k): v for k,v in inputs[0].items()}]
#     except:
#         True
    
#     try:
#         if isinstance(outputs, dict):
#             outputs = [{int(k): v for k,v in outputs.items()}]
#     except:
#         True

#     try:
#         if isinstance(outputs[0], dict):
#             outputs = [{int(k): v for k,v in outputs[0].items()}]
#     except:
#         True
    
#     return inputs, outputs

# print(f"len(inputs) = {len(inputs)}")
# inputs_list = []
# outputs_list = []
# for index, inputs in enumerate(inputs):
#     outputs = inout["outputs"][index]
#     inputs, outputs = process_input_output(inputs, outputs)
#     inputs_list.append(inputs)
#     outputs_list.append(outputs)

# print(f"inputs_list = {inputs_list} and len(inputs_list) = {len(inputs_list)} ")
# print(f"outputs_list = {outputs_list} and len(outputs_list) = {len(outputs_list)}")

# d_input = inputs_list[0].split("\n")
# d_output = outputs_list[0].split("\n")

# d_inputs = "\n".join(inputs_list)
# print(f"d_input = {d_input}")
# print(f"d_output = {d_output}")
# print(f"d_inputs = {d_inputs}")
# # for index, inputs in enumerate(inputs_list):
#     # print(f"index = {index}")
#     # print(f"inputs = {inputs}")
#     # inputs = inputs.strip().split("\n")
#     # print(f"inputs = {inputs}")
#     # n = int(inputs[0])
#     # print(f"n = {n}")
#     # for i in range(1, len(inputs), 2):
#     #     print(f"i = {i}")
#     #     m = int(inputs[i])
#     #     print(f"m = {m}")
#     #     for j in range(i+1, i+m+1):
#     #         print(f"j = {j}")

# code  = ["for _ in range(int(input())):\n n = int(input())\n mass = []\n zo = 0\n oz = 0\n zz = 0\n oo = 0\n ozs = []\n zos = []\n ozss = set()\n zoss = set()\n for j in range(n):\n k = input()\n mass.append(k)\n if k[0] == '0' and k[-1] == '1':\n zoss.add(k)\n zos.append(j + 1)\n zo += 1\n elif k[0] == '1' and k[-1] == '0':\n ozss.add(k)\n ozs.append(j + 1)\n oz += 1\n elif k[0] == '0' and k[-1] == '0':\n zz += 1\n else:\n oo += 1\n if zz and oo and not oz and not zo:\n print(-1)\n continue\n else:\n if zo > oz:\n print((zo - oz) // 2)\n ans = []\n need = (zo - oz) // 2\n i = 0\n while need:\n zzz = mass[zos[i] - 1][len(mass[zos[i] - 1]) - 1:: -1]\n if zzz not in ozss:\n ans.append(zos[i])\n need -= 1\n i += 1\n print(*ans)\n else:\n print((oz - zo) // 2)\n ans = []\n need = (oz - zo) // 2\n i = 0\n while need:\n zzz = mass[ozs[i] - 1][len(mass[ozs[i] - 1]) - 1:: -1]\n if zzz not in zoss:\n ans.append(ozs[i])\n need -= 1\n i += 1\n print(*ans)\n", "k = int(input())\nfor i in range(k):\n is_t = set()\n a = dict()\n a['00'] = []\n a['11'] = []\n a['01'] = []\n a['10'] = [] \n n = int(input())\n s = []\n for i in range(n):\n b = input()\n a[b[0] + b[-1]].append(i)\n s.append(b)\n is_t.add(b)\n c = len(a['10'])\n d = len(a['01'])\n if c + d == 0:\n if len(a['00']) == 0 or len(a['11']) == 0:\n print(0)\n else:\n print(-1)\n elif c > d:\n ans = []\n i = 0\n m = (d + c) // 2\n while d != m and i < len(a['10']):\n s1 = s[a['10'][i]]\n if s1[::-1] not in is_t:\n d += 1\n ans.append(a['10'][i] + 1)\n i += 1\n if d != m:\n print(-1)\n else:\n print(len(ans))\n print(*ans)\n else:\n ans = []\n i = 0\n m = (d + c) // 2\n while c != m and i < len(a['01']):\n s1 = s[a['01'][i]]\n if s1[::-1] not in is_t:\n c += 1\n ans.append(a['01'][i] + 1)\n i += 1\n if c != m:\n print(-1)\n else:\n print(len(ans))\n print(*ans)\n", "N = int(input())\n\ndef ceildiv(x, y):\n if x % y == 0:\n return x // y\n else:\n return x // y + 1\n\nfor _ in range(N):\n doms = []\n oc, zc = 0, 0\n n = int(input())\n\n used = set()\n fulls = dict()\n\n for i in range(n):\n d = input()\n used.add(d)\n if d[0] != d[-1]:\n fulls[i] = d\n doms.append((i, (d[0], d[-1])))\n else:\n if d[0] == '0':\n zc = 1\n else:\n oc = 1\n\n if len(doms) == 0:\n if zc == 1 and oc == 1:\n print(-1)\n else:\n print(0)\n else:\n # print(doms)\n\n _01 = 0\n _10 = 0\n\n _01_indexes = []\n _10_indexes = []\n\n\n for dom in doms:\n if dom[1] == ('0', '1'):\n _01 += 1\n _01_indexes.append(dom[0])\n else:\n _10 += 1\n _10_indexes.append(dom[0])\n\n if _10 < _01:\n _01, _10 = _10, _01\n _01_indexes, _10_indexes = _10_indexes, _01_indexes\n\n _10_indexes = [x for x in _10_indexes if fulls[x][::-1] not in used] \n\n need = ceildiv(_10-_01-1, 2)\n if len(_10_indexes) >= need:\n print(need)\n print( ' '.join(list([str(x+1) for x in _10_indexes[:need]])) )\n else:\n print(-1)\n\n # print(\"===\")\n # print(ceil(abs(doms.count(('0', '1')) - doms.count(('1', '0'))) - 1, 2))\n\n", "t=int(input())\nfor _ in range(t):\n n=int(input())\n k={\"01\":0,\"00\":0,\"11\":0,\"10\":0}\n ab=[]\n ba=[]\n a=[]\n ra=set()\n rb=set()\n for i in range(n):\n s=input()\n ts=s[0]+s[-1]\n k[ts]+=1\n if ts==\"01\":\n ab.append([str(i+1),s])\n ra.add(s)\n if ts==\"10\":\n ba.append([str(i+1),s])\n rb.add(s)\n if k[\"01\"]==0 and k[\"10\"]==0 and k[\"00\"]>0 and k[\"11\"]>0:\n ans=-1\n else:\n if k[\"01\"]==k[\"10\"] or k[\"01\"]==k[\"10\"]+1 or k[\"01\"]==k[\"10\"]-1:\n ans=0\n else:\n m=(k[\"01\"]+k[\"10\"])//2 if (k[\"01\"]+k[\"10\"])%2==0 else (k[\"01\"]+k[\"10\"])//2+1\n if k[\"01\"]>m:\n ans=k[\"01\"]-m\n for i in range(len(ab)):\n psp=ab[i][1]\n nn=list(psp)\n nn.reverse()\n psp=\"\".join(nn)\n c1=len(rb)\n rb.add(psp)\n c2=len(rb)\n if c1!=c2:\n a.append(ab[i][0])\n if len(a)>=ans:\n a=a[:ans]\n else:\n ans=-1\n else:\n ans=k[\"10\"]-m\n for i in range(len(ba)):\n psp=ba[i][1]\n nn=list(psp)\n nn.reverse()\n psp=\"\".join(nn)\n c1=len(ra)\n ra.add(psp)\n c2=len(ra)\n if c1!=c2:\n a.append(ba[i][0])\n if len(a)>=ans:\n a=a[:ans]\n else:\n ans=-1\n print(ans)\n if ans>0:\n print(\" \".join(a))\n", "t=int(input())\nfor i in range(t):\n n=int(input())\n i0,i1=[],[]\n l0,l1=[],[]\n h0,h1=False,False\n for i in range(n):\n t=input()\n if t[0]=='0' and t[-1]=='1':\n i0.append(i)\n l0.append(t)\n elif t[0]=='1' and t[-1]=='0':\n i1.append(i)\n l1.append(t)\n elif t[0]==t[-1]=='1':\n h1=True\n elif t[0]==t[-1]=='0':\n h0=True\n c0,c1=len(l0),len(l1)\n req,sl=0,[]\n s0=set(l0)\n s1=set(l1)\n if c0>0 or c1>0:\n if c0-c1>1:\n req=(c0-c1)//2\n sel=0\n sl=[]\n for tt in range(len(l0)):\n t=l0[tt]\n if not t[::-1] in s1:\n req-=1\n sl.append(i0[tt]+1)\n if req==0:\n break\n elif c1-c0>1:\n req=(c1-c0)//2\n sel=0\n sl=[]\n for tt in range(len(l1)):\n t=l1[tt]\n if not t[::-1] in s0:\n req-=1\n sl.append(i1[tt]+1)\n if req==0:\n break\n if req>0:\n print(-1)\n else:\n print(len(sl))\n print(*sl)\n else:\n if h0 and h1:\n print(-1)\n else:\n print(0)\n print(*[])\n"]

# # for co in code:
# #     print(co)
# code = "\n".join(code)

output_inputs = [ { "input": "20 40 60 80 100\n0 1 2 3 4\n1 0", "output": "4900" }, { "input": "119 119 119 119 119\n0 0 0 0 0\n10 0", "output": "4930" }, { "input": "3 6 13 38 60\n6 10 10 3 8\n9 9", "output": "5088" }, { "input": "21 44 11 68 75\n6 2 4 8 4\n2 8", "output": "4522" }, { "input": "16 112 50 114 68\n1 4 8 4 9\n19 11", "output": "5178" }, { "input": "55 66 75 44 47\n6 0 6 6 10\n19 0", "output": "6414" }, { "input": "47 11 88 5 110\n6 10 4 2 3\n10 6", "output": "5188" }, { "input": "5 44 61 103 92\n9 0 10 4 8\n15 7", "output": "4914" }, { "input": "115 53 96 62 110\n7 8 1 7 9\n7 16", "output": "3416" }, { "input": "102 83 26 6 11\n3 4 1 8 3\n17 14", "output": "6704" }, { "input": "36 102 73 101 19\n5 9 2 2 6\n4 13", "output": "4292" }, { "input": "40 115 93 107 113\n5 7 2 6 8\n6 17", "output": "2876" }, { "input": "53 34 53 107 81\n4 3 1 10 8\n7 7", "output": "4324" }, { "input": "113 37 4 84 66\n2 0 10 3 0\n20 19", "output": "6070" }, { "input": "10 53 101 62 1\n8 0 9 7 9\n0 11", "output": "4032" }, { "input": "45 45 75 36 76\n6 2 2 0 0\n8 17", "output": "5222" }, { "input": "47 16 44 78 111\n7 9 8 0 2\n1 19", "output": "3288" }, { "input": "7 54 39 102 31\n6 0 2 10 1\n18 3", "output": "6610" }, { "input": "0 46 86 72 40\n1 5 5 5 9\n6 5", "output": "4924" }, { "input": "114 4 45 78 113\n0 4 8 10 2\n10 12", "output": "4432" }, { "input": "56 56 96 105 107\n4 9 10 4 8\n2 1", "output": "3104" }, { "input": "113 107 59 50 56\n3 7 10 6 3\n10 12", "output": "4586" }, { "input": "96 104 9 94 84\n6 10 7 8 3\n14 11", "output": "4754" }, { "input": "98 15 116 43 55\n4 3 0 9 3\n10 7", "output": "5400" }, { "input": "0 26 99 108 35\n0 4 3 0 10\n9 5", "output": "5388" }, { "input": "89 24 51 49 84\n5 6 2 2 9\n2 14", "output": "4066" }, { "input": "57 51 76 45 96\n1 0 4 3 6\n12 15", "output": "5156" }, { "input": "79 112 37 36 116\n2 8 4 7 5\n4 12", "output": "3872" }, { "input": "71 42 60 20 7\n7 1 1 10 6\n1 7", "output": "5242" }, { "input": "86 10 66 80 55\n0 2 5 10 5\n15 6", "output": "5802" }, { "input": "66 109 22 22 62\n3 1 5 4 5\n10 5", "output": "5854" }, { "input": "97 17 43 84 58\n2 8 3 8 6\n10 7", "output": "5028" }, { "input": "109 83 5 114 104\n6 0 3 9 5\n5 2", "output": "4386" }, { "input": "94 18 24 91 105\n2 0 7 10 3\n1 4", "output": "4118" }, { "input": "64 17 86 59 45\n8 0 10 2 2\n4 4", "output": "5144" }, { "input": "70 84 31 57 2\n7 0 0 2 7\n12 5", "output": "6652" }, { "input": "98 118 117 86 4\n2 10 9 7 5\n11 15", "output": "4476" }, { "input": "103 110 101 97 70\n4 2 1 0 5\n7 5", "output": "4678" }, { "input": "78 96 6 97 62\n7 7 9 2 9\n10 3", "output": "4868" }, { "input": "95 28 3 31 115\n1 9 0 7 3\n10 13", "output": "5132" }, { "input": "45 17 116 58 3\n8 8 7 6 4\n3 19", "output": "3992" }, { "input": "19 12 0 113 77\n3 0 10 9 2\n8 6", "output": "5040" }, { "input": "0 0 0 0 0\n0 0 0 0 0\n0 0", "output": "7500" }, { "input": "0 0 0 0 0\n0 0 0 0 0\n20 0", "output": "9500" }, { "input": "119 119 119 119 119\n10 10 10 10 10\n0 20", "output": "1310" }, { "input": "0 0 0 0 0\n10 10 10 10 10\n0 20", "output": "4150" }, { "input": "119 0 0 0 0\n10 0 0 0 0\n5 5", "output": "7400" }, { "input": "0 119 0 0 0\n0 10 0 0 0\n5 5", "output": "7050" }, { "input": "0 0 119 0 0\n0 0 10 0 0\n0 0", "output": "6450" }, { "input": "0 0 0 119 0\n0 0 0 10 0\n5 5", "output": "6350" }, { "input": "0 0 0 0 119\n0 0 0 0 10\n5 5", "output": "6060" }, { "input": "119 0 0 0 0\n2 0 0 0 0\n5 5", "output": "7412" }, { "input": "0 119 0 0 0\n0 2 0 0 0\n5 5", "output": "7174" }, { "input": "0 0 119 0 0\n0 0 2 0 0\n5 5", "output": "6936" }, { "input": "0 0 0 119 0\n0 0 0 2 0\n5 5", "output": "6698" }, { "input": "0 0 0 0 119\n0 0 0 0 2\n5 5", "output": "6460" }, { "input": "119 0 0 0 0\n0 0 0 0 0\n4 9", "output": "7212" } ]
inputs_list = []
outputs_list = []
output_inputs = output_inputs[:1]
for index, input_output in enumerate(output_inputs):
    inputs = input_output["input"]
    outputs = input_output["output"]
    if isinstance(inputs, dict):
        inputs = [{int(k): v for k,v in inputs[0].items()}]
    if isinstance(outputs, dict):
        outputs = [{int(k): v for k,v in outputs[0].items()}]
    # outputs = output_input["output"][index]
    # inputs, outputs = process_input_output(input, outputs)

    print(f"inputs:\n{inputs}")
    print(f"outputs:\n{outputs}")
    d_inputs = "\n".join(inputs)
    d_output = "\n".join(outputs)
    print(f"d_input = {d_inputs}")
    print(f"d_output = {d_output}")
    # inputs_list.append(inputs)
    # outputs_list.append(outputs)