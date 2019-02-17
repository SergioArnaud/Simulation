mm1 <- function(lambdaA = 15, lambdaS = 10, n = 10){ 
  # Momento en que llega a la cola la i-ésima persona
  arribos = cumsum(rexp(n, 1/lambdaA))
  
  # Momento en que entra al servidor la i-ésima persona
  entrada = rep(0,n)
  entrada[1] = arribos[1]
  
  # Tiempo que tarda la i-ésima persona en salir del sistema (una vez que ya entró)
  tiempos_salida = rexp(n, 1/lambdaS)
  
  # Momento en que sale del servidor la i-ésima persona 
  salida = rep(0,n)
  salida[1] = arribos[1] + tiempos_salida[1]
  
  # Variable auxiliar que representa el número de personas esperando en la cola
  num_esperando = 0
  
  # i representa las llegadas, j las salidas. En este momento estamos esperando la llegada
  # del segundo y la salida del primero, por ello i=2, j=1
  i = 2; j = 1 
  
  # Arreglos auxiliares, representan los momentos en que la cola aumentó o disminuyo y el número 
  # de personas en la cola a partir de dicho momento.
  tiempos_cambio_cola = c(0)
  tamanos_cola = c(0)
  
  while(i<=n || j<n){
    
    # LLega gente a la cola y el servidor está ocupado
    while(i<=n && arribos[i] < salida[j]){
      
      num_esperando = num_esperando + 1
      tiempos_cambio_cola = append(tiempos_cambio_cola, arribos[i])
      tamanos_cola = append(tamanos_cola, num_esperando)
      
      i = i+1
    }
    
    # Sale alguien del servidor y hay gente en la cola
    if (num_esperando > 0){
      
      entrada[j+1] = salida[j]
      salida[j+1] = entrada[j+1] + tiempos_salida[j+1]
      
      num_esperando =  num_esperando - 1
      tiempos_cambio_cola = append(tiempos_cambio_cola, entrada[j+1])
      tamanos_cola = append(tamanos_cola, num_esperando)
      
      j = j+1 
    }
    
    # Sale alguien del servidor y no hay gente en la cola
    else{
      entrada[j+1] = arribos[i]
      salida[j+1] = entrada[j+1] + tiempos_salida[j+1]
      i = i+1; j = j+1
    }
  }
  
  # Tiempo de simulación (última salida del servidor)
  Tn = tail(salida, n=1)
  
  # Tiempo promedio de espera en la cola
  promedio_espera = sum(entrada-arribos) / (n)
  
  # Porcentaje de utilización del servidor
  utilizacion = 100 * sum(salida - entrada) / (Tn)
  
  # Promedio de tiempo en el sistema (cola + servidor)
  promedio_tiempo_en_sistema = sum(salida-arribos)/n
  
  # Longitud máxima alcanzada por la cola
  long_max_cola = max(tamanos_cola)
  
  # Máximo tiempo de espera en la cola
  max_espera = max(entrada-arribos)
  
  # El número promedio de clientes en la cola
  promedio_clientes = 0
  for(i in 2 : length(tiempos_cambio_cola)) {
    promedio_clientes = promedio_clientes + (tiempos_cambio_cola[i] - tiempos_cambio_cola[i-1])*(tamanos_cola[i-1])
  }
  promedio_clientes = promedio_clientes / (Tn)
  
  
  plot_df = data.frame(Tiempo = tiempos_cambio_cola, Tamano_cola = tamanos_cola)
  return (list( promedio_espera, utilizacion , promedio_clientes, promedio_tiempo_en_sistema, long_max_cola, max_espera, plot_df))
}