from Lab4.cifrados import lfsr, lgc, wichman
import math
import scipy.special as ss
from fractions import Fraction
import numpy as np
import random


#TEST TAKE FROM : https://github.com/GINARTeam/NIST-statistical-test
def test1(text, th=0.01):
    n = len(text)
    ones = text.count('1') #number of ones
    zeroes = text.count('0')    #number of zeros
    s = abs(ones - zeroes)
    p = math.erfc(float(s)/(math.sqrt(float(n)) * math.sqrt(2.0))) #p-value
    success = ( p >= th)  # success = true if p-value >= 0.01
    return [p, success]

def test2(text, th = 0.01, M=32):
    n = len(text)
    # Compute number of blocks M = block size. N=num of blocks
    # N = floor(n/M)
    # miniumum block size 20 bits, most blocks 100
    # fieldnames = ['number','chisq','p-value', 'success']

    N = int(math.floor(n/M))
    if N > 99:
        N=99
        M = int(math.floor(n/N))

    if n < 100:
        # Too little data for test. text of length at least 100 bits required
        return [0.0, 0.0, False]
    num_of_blocks = N
    block_size = M
    proportions = list()

    for i in range(num_of_blocks):
        block = text[i*(block_size):((i+1)*(block_size))]
        ones = block.count('1')
        zeroes = block.count('0')
        proportions.append(Fraction(ones,block_size))
    chisq = 0.0

    for prop in proportions:
        chisq += 4.0*block_size*((prop - Fraction(1,2))**2)
    p = ss.gammaincc((num_of_blocks/2.0),float(chisq)/2.0) # p-value
    success = (p>= th)
    return [p, success]

def test3(text, th = 0.01):
    n = len(text)
    ones = text.count('1') #number of ones
    zeroes = text.count('0')    #number of zeros
    prop = float(ones)/float(n)
    tau = 2.0/math.sqrt(n)
    vobs = 0.0

    if abs(prop-0.5) > tau:
        p = 0
    else:
        vobs = 1.0
        for i in range(n-1):
            if text[i] != text[i+1]:
                vobs += 1.0

        p = math.erfc(abs(vobs - (2.0*n*prop*(1.0-prop)))/(2.0*math.sqrt(2.0*n)*prop*(1-prop) ))
    success = (p>=th)
    return [p, success]

def test4(text, th = 0.01):
    n = len(text)
    M8 = [0.2148, 0.3672, 0.2305, 0.1875]
    # Length of blocks
    M = 8
    K = 3
    N = 16
    # Table of frequencies
    v = [0,0,0,0,0,0,0]

    for i in range(N): # over each block
        #find the longest run
        block = text[i*M:((i+1)*M)] # Block i
        run = 0
        longest = 0
        for j in range(M): # Count the bits.
            if block[j] == '1':
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0

        if longest <= 1:    v[0] += 1
        elif longest == 2:  v[1] += 1
        elif longest == 3:  v[2] += 1
        else:               v[3] += 1
    # Compute Chi-Sq
    chi_sq = 0.0
    for i in range(K+1):
        p_i = M8[i]
        upper = (v[i] - N*p_i)**2
        lower = N*p_i
        chi_sq += upper/lower
    # p-value
    p = ss.gammaincc(K/2.0, chi_sq/2.0)
    success = (p>=th)
    return [p, success]

#5 no jalo

#este es el 6
def test5(text, th = 0.01):
    n = len(text)
    T = math.sqrt(math.log(1.0/0.05)*n) # Compute upper threshold
    N0 = 0.95*n/2.0
    write_array = [0.0,0.0,0.0,0.0]
    ts = list()             # Convert to +1,-1
    for i in range(n):
        if text[i] == '1':
            ts.append(1)
        else:
            ts.append(-1)
    ts_np = np.array(ts)

    fs = np.fft.fft(ts_np)  # Compute DFT

    mags = abs(fs)[:int(n/2)]  #Compute magnitudes of first half of sequence

    N1 = 0.0   # Count the peaks above the upper theshold
    for mag in mags:
        if mag < T:
            N1 += 1.0
    d = (N1 - N0)/math.sqrt((n*0.95*0.05)/4)
    # Compute the P value
    p = math.erfc(abs(d)/math.sqrt(2))
    success = (p>=th)
    return [p, success]

#este es el 7
def test6(text, th = 0.01):
    n = len(text)
    # The templates provdided in SP800-22rev1a
    templates = [None for x in range(7)]
    templates[0] = [[0,1],[1,0]]
    templates[1] = [[0,0,1],[0,1,1],[1,0,0],[1,1,0]]
    templates[2] = [[0,0,0,1],[0,0,1,1],[0,1,1,1],[1,0,0,0],[1,1,0,0],[1,1,1,0]]
    templates[3] = [[0,0,0,0,1],[0,0,0,1,1],[0,0,1,0,1],[0,1,0,1,1],[0,0,1,1,1],[0,1,1,1,1],
                    [1,1,1,0,0],[1,1,0,1,0],[1,0,1,0,0],[1,1,0,0,0],[1,0,0,0,0],[1,1,1,1,0]]
    templates[4] = [[0,0,0,0,0,1],[0,0,0,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,1],[0,0,1,0,1,1],
                    [0,0,1,1,0,1],[0,0,1,1,1,1],[0,1,0,0,1,1],
                    [0,1,0,1,1,1],[0,1,1,1,1,1],[1,0,0,0,0,0],
                    [1,0,1,0,0,0],[1,0,1,1,0,0],[1,1,0,0,0,0],
                    [1,1,0,0,1,0],[1,1,0,1,0,0],[1,1,1,0,0,0],
                    [1,1,1,0,1,0],[1,1,1,1,0,0],[1,1,1,1,1,0]]
    templates[5] = [[0,0,0,0,0,0,1],[0,0,0,0,0,1,1],[0,0,0,0,1,0,1],[0,0,0,0,1,1,1],
                    [0,0,0,1,0,0,1],[0,0,0,1,0,1,1],[0,0,0,1,1,0,1],[0,0,0,1,1,1,1],
                    [0,0,1,0,0,1,1],[0,0,1,0,1,0,1],[0,0,1,0,1,1,1],[0,0,1,1,0,1,1],
                    [0,0,1,1,1,0,1],[0,0,1,1,1,1,1],[0,1,0,0,0,1,1],[0,1,0,0,1,1,1],
                    [0,1,0,1,0,1,1],[0,1,0,1,1,1,1],[0,1,1,0,1,1,1],[0,1,1,1,1,1,1],
                    [1,0,0,0,0,0,0],[1,0,0,1,0,0,0],[1,0,1,0,0,0,0],[1,0,1,0,1,0,0],
                    [1,0,1,1,0,0,0],[1,0,1,1,1,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,1,0],
                    [1,1,0,0,1,0,0],[1,1,0,1,0,0,0],[1,1,0,1,0,1,0],[1,1,0,1,1,0,0],
                    [1,1,1,0,0,0,0],[1,1,1,0,0,1,0],[1,1,1,0,1,0,0],[1,1,1,0,1,1,0],
                    [1,1,1,1,0,0,0],[1,1,1,1,0,1,0],[1,1,1,1,1,0,0],[1,1,1,1,1,1,0]]
    templates[6] = [[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,1],[0,0,0,0,0,1,0,1],[0,0,0,0,0,1,1,1],
                    [0,0,0,0,1,0,0,1],[0,0,0,0,1,0,1,1],[0,0,0,0,1,1,0,1],[0,0,0,0,1,1,1,1],
                    [0,0,0,1,0,0,1,1],[0,0,0,1,0,1,0,1],[0,0,0,1,0,1,1,1],[0,0,0,1,1,0,0,1],
                    [0,0,0,1,1,0,1,1],[0,0,0,1,1,1,0,1],[0,0,0,1,1,1,1,1],[0,0,1,0,0,0,1,1],
                    [0,0,1,0,0,1,0,1],[0,0,1,0,0,1,1,1],[0,0,1,0,1,0,1,1],[0,0,1,0,1,1,0,1],
                    [0,0,1,0,1,1,1,1],[0,0,1,1,0,1,0,1],[0,0,1,1,0,1,1,1],[0,0,1,1,1,0,1,1],
                    [0,0,1,1,1,1,0,1],[0,0,1,1,1,1,1,1],[0,1,0,0,0,0,1,1],[0,1,0,0,0,1,1,1],
                    [0,1,0,0,1,0,1,1],[0,1,0,0,1,1,1,1],[0,1,0,1,0,0,1,1],[0,1,0,1,0,1,1,1],
                    [0,1,0,1,1,0,1,1],[0,1,0,1,1,1,1,1],[0,1,1,0,0,1,1,1],[0,1,1,0,1,1,1,1],
                    [0,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0],[1,0,0,1,0,0,0,0],[1,0,0,1,1,0,0,0],
                    [1,0,1,0,0,0,0,0],[1,0,1,0,0,1,0,0],[1,0,1,0,1,0,0,0],[1,0,1,0,1,1,0,0],
                    [1,0,1,1,0,0,0,0],[1,0,1,1,0,1,0,0],[1,0,1,1,1,0,0,0],[1,0,1,1,1,1,0,0],
                    [1,1,0,0,0,0,0,0],[1,1,0,0,0,0,1,0],[1,1,0,0,0,1,0,0],[1,1,0,0,1,0,0,0],
                    [1,1,0,0,1,0,1,0],[1,1,0,1,0,0,0,0],[1,1,0,1,0,0,1,0],[1,1,0,1,0,1,0,0],
                    [1,1,0,1,1,0,0,0],[1,1,0,1,1,0,1,0],[1,1,0,1,1,1,0,0],[1,1,1,0,0,0,0,0],
                    [1,1,1,0,0,0,1,0],[1,1,1,0,0,1,0,0],[1,1,1,0,0,1,1,0],[1,1,1,0,1,0,0,0],
                    [1,1,1,0,1,0,1,0],[1,1,1,0,1,1,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,1,0],
                    [1,1,1,1,0,1,0,0],[1,1,1,1,0,1,1,0],[1,1,1,1,1,0,0,0],[1,1,1,1,1,0,1,0],
                    [1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0]]

    # Randomly choose the template B
    r = random.SystemRandom()
    template_list = r.choice(templates)
    B = r.choice(template_list)

    m = len(B)

    N = 8  #number of block
    M = int(n/N)    #length of each block

    blocks = list() # Split into N blocks of M bits
    for i in range(N):
        block = list()
        for j in range(M):
            block.append(int(text[i*M+j],2))
        blocks.append(block)

    W=list() # Count the number of matches of the template in each block Wj
    for block in blocks:
        position = 0
        count = 0
        while position < (M-m):

            if block[position:position+m] == B:
                position += m
                count += 1
            else:
                position += 1
        W.append(count)

    mu = float(M-m+1)/float(2**m) # Compute mu and sigma
    sigma = M * ((1.0/float(2**m))-(float((2*m)-1)/float(2**(2*m))))

    chi_sq = 0.0  # Compute Chi-Square
    for j in range(N):
        chi_sq += ((W[j] - mu)**2)/sigma
    p = ss.gammaincc(N/2.0, chi_sq/2.0) # Compute P value
    success = ( p >= th)
    return [p, success]

#este es el 9
def test7(text, th = 0.01, patternlen=None, initblocks=None):
    n = len(text)
    # Step 1. Choose the block size
    if patternlen != None:
        L = patternlen
    else:
        ns = [904960,2068480,4654080,10342400,
              22753280,49643520,107560960,
              231669760,496435200,1059061760]
        L = 6
        if n < 387840:
            # Too little data. texts of length at least 387840 are recommended
            return [0] * 8
        for threshold in ns:
            if n >= threshold:
                L += 1

    # Step 2 Split the data into Q and K blocks
    nblocks = int(math.floor(n/L))
    if initblocks != None:
        Q = initblocks
    else:
        Q = 10*(2**L)
    K = nblocks - Q

    # Step 3 Construct Table
    nsymbols = (2**L)
    T=[0 for x in range(nsymbols)] # zero out the table
    for i in range(Q):             # Mark final position of
        pattern = text[i*L:(i+1)*L] # each pattern
        idx = int(pattern, 2)
        T[idx]=i+1      # +1 to number indexes 1..(2**L)+1
                        # instead of 0..2**L
    # Step 4 Iterate
    sum = 0.0
    for i in range(Q,nblocks):
        pattern = text[i*L:(i+1)*L]
        j = int(pattern,2)
        dist = i+1-T[j]
        T[j] = i+1
        sum = sum + math.log(dist,2)

    # Step 5 Compute the test statistic
    fn = sum/K

    # Step 6 Compute the P Value
    # Tables from https://static.aminer.org/pdf/PDF/000/120/333/
    # a_universal_statistical_test_for_random_bit_generators.pdf
    ev_table =  [0,0.73264948,1.5374383,2.40160681,3.31122472,
                 4.25342659,5.2177052,6.1962507,7.1836656,
                 8.1764248,9.1723243,10.170032,11.168765,
                 12.168070,13.167693,14.167488,15.167379]
    var_table = [0,0.690,1.338,1.901,2.358,2.705,2.954,3.125,
                 3.238,3.311,3.356,3.384,3.401,3.410,3.416,
                 3.419,3.421]

    # sigma = math.sqrt(var_table[L])
    sigma = abs((fn - ev_table[L])/((math.sqrt(var_table[L]))*math.sqrt(2)))
    P = math.erfc(sigma)

    success = (P >= th)
    return [P, success]

#este es el 12
def test8(text, th = 0.01):
    n = len(text)
    m = int(math.floor(math.log(n,2)))-6
    if m < 2:
        m = 2
    if m >3 :
        m = 3

    Cmi = list()
    phi_m = list()
    for iterm in range(m,m+2):
        # Step 1
        padded_text=text+text[0:iterm-1]

        # Step 2
        counts = list()
        for i in range(2**iterm):
            count = 0
            for j in range(n):
                if int(padded_text[j:j+iterm],2) == i:
                    count += 1
            counts.append(count)

        # step 3
        Ci = list()
        for i in range(2**iterm):
            Ci.append(float(counts[i])/float(n))

        Cmi.append(Ci)

        # Step 4
        sum = 0.0
        for i in range(2**iterm):
            if (Ci[i] > 0.0):
                sum += Ci[i]*math.log((Ci[i]/10.0))
        phi_m.append(sum)

    # Step 5 - let the loop steps 1-4 complete

    # Step 6
    appen_m = phi_m[0] - phi_m[1]

    chisq = 2*n*(math.log(2) - appen_m)

    # Step 7
    p = ss.gammaincc(2**(m-1),(chisq/2.0))

    success = (p >= th)

    return [p, success]

#test 14
def test9(text, th = 0.01):
    n = len(text)
    # Convert to +1,-1
    x = list()
    for i in range(n):
        x.append(int(text[i],2)*2 -1 )

    # Build the partial sums
    pos = 0
    s = list()
    for e in x:
        pos = pos+e
        s.append(pos)
    sprime = [0]+s+[0] # Add 0 on each end


    # Build the list of cycles
    pos = 1
    cycles = list()
    while (pos < len(sprime)):
        cycle = list()
        cycle.append(0)
        while sprime[pos]!=0:
            cycle.append(sprime[pos])
            pos += 1
        cycle.append(0)
        cycles.append(cycle)
        pos = pos + 1

    J = len(cycles)

    vxk = [['a','b','c','d','e','f'] for y in [-4,-3,-2,-1,1,2,3,4] ]

    # Count Occurances
    for k in range(6):
        for index in range(8):
            mapping = [-4,-3,-2,-1,1,2,3,4]
            x = mapping[index]
            cyclecount = 0
            #count how many cycles in which x occurs k times
            for cycle in cycles:
                oc = 0
                #Count how many times x occurs in the current cycle
                for pos in cycle:
                    if (pos == x):
                        oc += 1
                # If x occurs k times, increment the cycle count
                if (k < 5):
                    if oc == k:
                        cyclecount += 1
                else:
                    if k == 5:
                        if oc >=5:
                            cyclecount += 1
            vxk[index][k] = cyclecount

    # Table for reference random probabilities
    pikx=[[0.5     ,0.25   ,0.125  ,0.0625  ,0.0312 ,0.0312],
          [0.75    ,0.0625 ,0.0469 ,0.0352  ,0.0264 ,0.0791],
          [0.8333  ,0.0278 ,0.0231 ,0.0193  ,0.0161 ,0.0804],
          [0.875   ,0.0156 ,0.0137 ,0.012   ,0.0105 ,0.0733],
          [0.9     ,0.01   ,0.009  ,0.0081  ,0.0073 ,0.0656],
          [0.9167  ,0.0069 ,0.0064 ,0.0058  ,0.0053 ,0.0588],
          [0.9286  ,0.0051 ,0.0047 ,0.0044  ,0.0041 ,0.0531]]

    success = True
    plist = list()
    chi_sq = list()
    p_total = 0.0
    for index in range(8):
        #list of states
        mapping = [-4,-3,-2,-1,1,2,3,4]
        x = mapping[index]
        chisq = 0.0
        for k in range(6):
            top = float(vxk[index][k]) - (float(J) * (pikx[abs(x)-1][k]))
            top = top*top
            bottom = J * pikx[abs(x)-1][k]
            chisq += top/bottom

        p = ss.gammaincc(5.0/2.0,chisq/2.0)
        p_total += p
        plist.append(p)
        chi_sq.append(chisq)
        if p < th:
            success = False

    return [p_total/8, success]

#este es el 15
def test10(text, th = 0.01):
    n = len(text)
    x = list()             # Convert to +1,-2
    for i in range(n):
        x.append(int(text[i],2)*2-1)

    # Build the partial sums
    pos = 0
    s = list()
    for e in x:
        pos = pos+e
        s.append(pos)
    # print(s)
    sprime = [0]+s+[0] # Add 0 on each end

    # Count the number of cycles J
    J = 0
    for value in sprime[1:]:
        if value == 0:
            J += 1

    # Build the counts of offsets
    count = [0 for x in range(-9,10)]
    for value in sprime:
        if (abs(value) < 10):
            count[value] += 1

    # Compute P values
    success = True
    plist = list() # list of p-values for each state
    p_average = 0.0
    for x in range(-9,10):
        if x != 0:
            top = abs(count[x]-J)
            bottom = math.sqrt(2.0 * J *((4.0*abs(x))-2.0))
            p = ss.erfc(top/bottom)

            # print("p[" + str(x) +"] = " + str(p))

            p_average +=p
            plist.append(p)
            if p < th:
                success = False

    return [p, success]

#TODO REVISAR EN EL METODO QUE FALLA PARA VER SI DE VERDAD FALLA
def successTable(bits):

    for i in range(10):
        result = globals()['test{}'.format(i+1)](bits)
        print('Test ',(i+1),', p = ',result[0],', result = ',result[1])


#to generate histogram
#TODO Revisar que cosas van random y hacer el histograma
def generateLGC():
    for i in range(200):
        a  = random.randint(0,40000)
        b = random.randint(0, 40000)
        N = random.randint(0,40000)
        bits = lgc(a,b,N,35,100)
        successTable(bits)

def generateWichman():
    for i in range(200):
        s1 = random.randint(0,30000)
        s2 = random.randint(0,30000)
        s3 = random.randint(0,30000)

        bits = wichman(s1,s2,s3,100)
        successTable(bits)

def generateLFSR():
    for i in range(200):
        bits = lfsr(seed=format('110100011011011010111101'), taps=[2,5,7,10,15,11], nbits=100)
        successTable(bits)