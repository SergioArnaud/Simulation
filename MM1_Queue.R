mm1_s <- function(lambdaA = 15, lambdaS = 10, n = 100){ 
    arribos = cumsum(rexp(n, 1/lambdaA))
    tiempos_salida = rexp(n, 1/lambdaS)

    entrada = rep(0,n)
    entrada[1] = arribos[1]

    salida = rep(0,n)
    salida[1] = arribos[1] + tiempos_salida[1]

    num_esperando = 0

    i = 2 
    j = 1 

    df <- data.frame(0,0)
    names(df)<-c("Tiempo","Tamano cola")

    while(i<=n || j<n){

        while(i<=n && arribos[i] < salida[j]){

            num_esperando = num_esperando + 1
            df[nrow(df) + 1,] = list(arribos[i],num_esperando)
            i = i +1
        }

        if (num_esperando > 0){

            entrada[j+1] = salida[j]
            salida[j+1] = entrada[j+1] + tiempos_salida[j+1]

            num_esperando =  num_esperando - 1
            df[nrow(df) + 1,] = list(entrada[j+1],num_esperando)
            j = j + 1 
        }

        else{

            entrada[j+1] = arribos[i]
            salida[j+1] = entrada[j+1] + tiempos_salida[j+1]

            i = i+1
            j = j+1
        }

    }

    Tn = tail(salida, n=1)
    return (list( sum(entrada-arribos)/n, 100 * sum(salida - entrada) / Tn, df))
}


df <- data.frame(0,0)
names(df)<-c("Promedio_espera","Utilizacion")

for (i in 1:10){
    ans = mm1_s(lambdaA = 2, lambdaS = 1, n = 1000)[1:2]
    df[nrow(df) + 1,] = c(ans[1],ans[2])
}

mean(df$Promedio_espera)
mean(df$Utilizacion)


dd = mm1_s(lambdaA = 2, lambdaS = 1, n = 1000)[3]
plot(data.frame(dd) ,  type='s')
