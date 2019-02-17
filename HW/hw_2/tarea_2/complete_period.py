from sympy.ntheory import primefactors, factorint, isprime
import math

def gcd(a, b):
    if b > a:
        return gcd(b, a)
    return b if a % b == 0 else gcd(b, a % b)

def are_coprime(a,b):
    mcd = gcd(a,b)

    if mcd != 1:
        print('Los nums. no son primos relativos,\
              su MCD({},{}) = {}'.format(a,b,mcd))

    return mcd == 1

def verify_condition_2(m,a):
    prime_factors = primefactors(int(m/2))
    for prime_factor in prime_factors:
        if (a-1)%prime_factor != 0:
            print('Falla cond. 2:\n{} es primo\
            y divide a m={} pero no a (a-1)={}'
            .format(prime_factor, m, a-1))
            return False
    return True

def verify_condition_3(m,a):
    if m%4 == 0 and (a-1)%4 != 0:
        print('Falla cond. 3:\n4 divide a\
        m={} pero no a (a-1)={}'.format( m, a-1))
        return False
    return True

def complete_period(a,c,m):
    
    coprime     = are_coprime(c,m)
    condition_2 = verify_condition_2(m,a)
    condition_3 = verify_condition_3(m,a)
    
    if coprime and condition_2 and condition_3:
        return True
    return False
    