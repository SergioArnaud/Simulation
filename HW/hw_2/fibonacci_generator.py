import math

def fibo(x0, x1, m=5):
    s = set(); s.add((x0,x1))
    sequence = [str(x0)]
    
    val = (x0+x1) % m
    x0 = x1; x1 = val
    
    while not (x0,x1) in s:
        s.add((x0,x1))
        sequence.append(str(x0))
        
        val = (x0+x1) % m
        x0 = x1; x1 = val
    
    return sequence

def get_different_cicles_fibonacci(n=5):
    different_cicles = []

    for x0 in range(n):
        for x1 in range(n):

            new_cicle = fibo(x0,x1,n)
            new_cicle_s = '-'.join(new_cicle)+'-'

            is_new = True

            for cicle in different_cicles:
                cicle_s = '-'.join(cicle)+'-'

                if (cicle_s in (new_cicle_s*2)) and (new_cicle_s in (cicle_s*2)):
                    is_new = False
                    break

            if is_new:
                different_cicles.append(new_cicle)
    return different_cicles