---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.2
---

```python
import pymc3 as pm
import seaborn as sn
import numpy as np
import scipy
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from scipy import stats
from astropy.stats import bootstrap
import seaborn as sns
```

Sergio Arnaud Gómez

Jorge Rotter Vallejo


## Pregunta 1


1.$ Y|\theta \sim G(1,\theta)$ y $\theta \sim IG(\alpha, \beta)$

```python
alpha = 6
beta  = 1
with pm.Model() as model:
    theta = pm.InverseGamma('theta',alpha=alpha, beta=beta)
    Y = pm.Gamma('y',1,theta)
    trace = pm.sample(1000)
pm.traceplot(trace)
```

- Encuentren la distribución posterior de $\theta.$

```python
pm.plot_posterior(trace['theta'], figsize=(12,5))
```

Encuentren la media y varianza posterior de θ.

```python
'Media: ', np.mean(trace['theta'])
```

```python
'Varianza: ',  np.var(trace['theta'])
```

Encuentren la moda posterior de θ.

```python
map_estimate = pm.find_MAP(model=model)
map_estimate['theta']
```

Escriban dos ecuaciones integrales que se pueden resolver para encontrar el intervalo de 95 % de colas simétricas para θ.


## Pregunta 2


Los siguientes datos corresponden a las horas adicionales de sueño de 10 pacientes tratados con un somnífero B, comparado con un somnífero A:

```python
x = np.array([1.2, 2.4, 1.3, 1.3, 0, 1, 1.8, 0.8, 4.6, 1.4])
```

Presentamos tres análisis bayesiano de estos datos, suponiendo diferentes verosimilitudes. En todos los casos usamos  distribuciones iniciales normales $\mathcal{N}(0, 1)$ para $\mu$ y $\mathrm{HalfCauchy}(1)$ para $\sigma^2$.

```python
mu_sp = stats.norm.rvs(size=1000)
sns.distplot(mu_sp)
```

```python
ssq_sp = stats.halfcauchy.rvs(size=1000)
sns.distplot(ssq_sp[30 > ssq_sp])
```

### a. $X|\theta \sim \mathcal{N}$

```python
with pm.Model() as modelo_p2a:
    # Iniciales
    mu = pm.Normal('mu', 0,1)
    sigma_sq = pm.HalfCauchy('sigma_sq', 1)
    
    # Verosimilitud
    treatment_normal = pm.Normal('normal', mu, sigma_sq, observed=x)
    
    # Muestreo
    trace = pm.sample(start=pm.find_MAP())
    
```

```python
pm.traceplot(trace)
```

### b. $X|\theta \sim t_{(3)}$.

```python
with pm.Model() as modelo_p2b:
    # Iniciales
    mu = pm.Normal('mu', 0,1)
    
    # Verosimilitud
    treatment_t3 = pm.StudentT('t3', nu=3, mu=mu, observed=x)
    
    # Muestreo
    trace = pm.sample(start=pm.find_MAP())
    
```

```python
pm.traceplot(trace)
```

Aquí la distribución inicial de sigma fue demasiado laxa.


### c. $X|\theta \sim t_{(1)}$

```python
with pm.Model() as modelo_p2c:
    # Iniciales
    mu = pm.Normal('mu', 0,1)
    
    # Verosimilitud
    treatment_t1 = pm.StudentT('t1', nu=1, mu=mu, observed=x)
    
    # Muestreo
    trace = pm.sample(start=pm.find_MAP())
    
```

```python
pm.traceplot(trace)
```

### d. $Y = \left[X<\bar{X}\right] \ , \ Y|\theta \sim \textrm{Bernoulli}(\theta)$

```python
with pm.Model() as modelo_p2a:
    # Iniciales
    theta = pm.Uniform('theta', lower=0, upper=1)
    
    # Verosimilitud
    treatment_blli = pm.Bernoulli('blli', p=theta, observed=np.array([x>np.mean(x)])*1)
    
    # Muestreo
    trace = pm.sample(start=pm.find_MAP(), tune=1000)
    
```

```python
pm.traceplot(trace)
```

## Pregunta 3


Especificación de los datos

```python
arr = np.array([1.6907,6,59,1.7242,13,60,1.7552, 18,62, 
                1.7842,28,56,1.8113,52,63, 1.8369,53,59, 
                1.8610,61,62,1.8839,60,60])
arr = np.reshape(arr, (8,3))
w = arr[:,0]
y = arr[:,1]
n = arr[:,2]
```

Con función logit

```python
with pm.Model() as model:
    alpha = pm.Normal('alpha',0,sd=1/.001)
    beta = pm.Normal('beta',0,sd=1/.001)
    p = pm.Deterministic('p',pm.invlogit(alpha + beta*(w - np.mean(w))))
    deaths = pm.Binomial('deaths', n=n, p=p, observed=y)
    trace = pm.sample()
pm.traceplot(trace)
```

```python
pm.plot_posterior(trace)
```

Con función probit

```python
with pm.Model() as model:
    alpha = pm.Normal('alpha',0,sd=1/.001)
    beta = pm.Normal('beta',0,sd=1/.001)
    p = pm.Deterministic('p',pm.invprobit(alpha + beta*(w - np.mean(w))))
    deaths = pm.Binomial('deaths', n=n, p=p, observed=y)
    trace = pm.sample()
pm.traceplot(trace)
```

```python
pm.plot_posterior(trace)
```

Con función complementaria log-log

```python
def cloglog(x):
    return 1 - np.exp(-np.exp(x))

with pm.Model() as model:
    alpha = pm.Normal('alpha',0,sd=1/.001)
    beta = pm.Normal('beta',0,sd=1/.001)
    p = pm.Deterministic('p',cloglog(alpha + beta*(w - np.mean(w))))
    deaths = pm.Binomial('deaths', n=n, p=p, observed=y)
    trace = pm.sample()
pm.traceplot(trace)
```

```python
pm.plot_posterior(trace)
```

## Pregunta 4


Considere las siguientes dos distribuciones condicionales completas, analizadas en el artículo de Casella y George (1992):

$$
f(x|y) \propto ye^{-yx}, \ \ \ \ 0<x<B \\
f(y|x) \propto xe^{-xy}, \ \ \ \ 0<y<B
$$

Obtenga un estimado de la distribución marginal de $X$ cuando $B=10$ usando Gibbs sampler.


## Pregunta 5


En una prueba real, 12 lotes de mantequilla de cacahuate tienen residuos de aflato- xin en partes por mil millones de:

```python
arr = np.array([4.94, 5.06, 4.53, 5.07, 4.99, 5.16, 4.38, 4.43, 4.93, 4.72, 4.92, 4.96])
```

- ¿Cuántas posibles muestras boototrap hay en estos datos

```python
n = len(arr)
scipy.special.comb(2*n-1,n-1,exact=True)
```

Usando R y la función sample, o una tabla de números aleatorios, generar 100 remuestras de los datos de la muestra. Para cada una de estas remuestras, obtener la media. Comparar la media de las medias obtenidas en las remuestras con la media de la muestra original.

```python
sample = np.random.choice(arr, size=100, replace=True)
np.mean(sample)
```

```python
np.mean(arr)
```

Y comparando

```python
err = 100*(np.mean(sample) - np.mean(arr))/ np.mean(arr)
print('Erorr = {}%'.format(round(err,4)))
```

Encontrar de las 100 remuestras, un intervalo de confianza del 95 % para la media.

```python
ans = bs.bootstrap(arr,bs_stats.mean,num_iterations=100, alpha=.05)
print('Intervalo de confianza al nivel 95%: [{},{}]'.format(round(ans.lower_bound,2), round(ans.upper_bound,2)))
```

## Pregunta 6

El número de accidentes aéreoes de 1983 a 2006 está dado en el vector

```python
x = np.array([23, 16, 21, 24, 34, 30, 28, 24, 26, 18, 23, 36, 37, 49, 50, 51, 56, 46, 41, 54, 30, 40, 40, 31])
```

Calcule la media (con su error estándar) y la mediana.

```python
np.mean(x)
```

```python
np.std(x)/np.sqrt(len(x))
```

```python
np.median(x)
```

Calcule estimaciones bootstrap de la media y la mediana, ambas con sus errores estándar, usando $B=1000$ remuestras. Calcule la mediana de las medianas muestrales.

```python
B = 1000

# Medias boostrap
medias = bootstrap(x, B, bootfunc=np.mean)
print(np.mean(medias))
sns.distplot(medias)
```

```python
# Error estándar del estimador bootstrap de la media
def bootstrap_se(x):
    return np.std(x)/np.sqrt(len(x))

bootstrap_se(medias)
```

```python
# Medianas boostrap
medianas = bootstrap(x, B, bootfunc=np.median)
print(np.median(medianas))
sns.distplot(medianas)
```

```python
# Error estándar del estimador mediana bootstrap
bootstrap_se(medianas)
```

La mediana de las medianas coincide con la estimación plana de la mediana, mientras que la media baja (pero muy poco). En erro estándar, el de la media es menor que el de la mediana.


## Pregunta 7


La $\tau$ de Kendall entre $X$ y $Y$ es $0.55$ y tanto $X$ como $Y$ son positivas. 


¿Cuál es la $\tau$ entre $X$ y $\frac{1}{Y}$?
¿Cuál es la $\tau$ de $\frac{1}{X}$ y $\frac{1}{Y}$?


Recordemos que la $\tau$ de Kendall está dada por la siguiente expresión:

$$ \rho_{\tau}(X,Y) = P((X −X^*)(Y −Y^*) > 0)−P((X −X^*)(Y −Y^*) < 0) $$


De hecho es fácil notar que si $U=F(X)$ y $V=G(Y)$: 

$$ \begin{eqnarray}  \rho_{\tau}(X,Y) &=& P((X −X^*)(Y −Y^*) > 0)−P((X −X^*)(Y −Y^*) < 0) \\ &=& 4P(X<X^*,Y<Y^*) - 1 \\ &=& 4P(F(X)<F(X^*),G(Y)<G(Y^*) - 1\\ &=& 4P(U<U^*,V<V^*) - 1 \\ \end{eqnarray} $$
    
De hecho:

$$ \rho_{\tau}(X,Y) = 4 \int_0^1 \int_0^1 C_{X,Y}(u,v) d_{C_{X,Y}(u,v)} -1 $$ 


Donde C_{X,Y} es la cópula entre X y Y. Concluimos que la tau de kendal es invariante ante transformaciones monótonas. Dado que ambas son positivas, $\frac{1}{Y}$ es monótona decreciente de forma que el valor de la $\tau$ entre $X$ y $\frac{1}{Y}$ es -.55. Siguiendo el mismo argumento, el valor de la $\tau$ de $\frac{1}{X}$ y $\frac{1}{Y}$ es .55
