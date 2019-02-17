mm1 <- function(lambdaA = 15, lambdaS = 10, n = 100){
  inicializa(lambdaA = lambdaA, lambdaS = lambdaS, n = n) 
  while (clientes_enespera < n){
    tiempo()
    actualiza_estadisticas() 
    if(sig_tipo_evento == 1) llegadas() else salidas() 
    next
  }
  return(reporte()) 
}

inicializa  <- function(lambdaA, lambdaS, n){
  lambdaA  <<- lambdaA   
  lambdaS  <<- lambdaS   
  n        <<- n         
  num_eventos <<- 2       
  lt_A     <<- vector(mode="numeric", length = 1) 
  reloj    <<- 0             
  servidor <<- 0          
  q_t      <<- 0 
  tiempo_ultimo_evento <<- 0
  
  clientes_enespera <<- 0
  total_esperas     <<- 0
  area_q            <<- 0
  area_status_servidor <<- 0
  
  tiempo_sig_evento <<- c(reloj + rexp(1, 1/lambdaA), 1e30)
  le <<- c(e=reloj,tipo=0,q=q_t)
}


tiempo <- function(){
  min_tiempo_sig_evento <<- 1e29  
  sig_tipo_evento <<- 0

  for(i in 1:num_eventos){
    if( tiempo_sig_evento[i] < min_tiempo_sig_evento ){
      min_tiempo_sig_evento <<- tiempo_sig_evento[i]
      sig_tipo_evento <<- i
    }
  }
  
  if(sig_tipo_evento == 0){
    stop(print(paste("La lista de eventos está vacía en el tiempo:", reloj, sep=" ")))
  }
  
  reloj <<- min_tiempo_sig_evento
  le <<- rbind(le,c(reloj,sig_tipo_evento,q=q_t))
}


llegadas <- function(){
  tiempo_sig_evento[1] <<- reloj + rexp(1, 1/lambdaA) 
  if(servidor == 1){
    q_t <<- q_t + 1 
    lt_A[q_t] <<- reloj  
  } else {
    Di <<- 0
    total_esperas <<- total_esperas + Di
    clientes_enespera <<- clientes_enespera + 1
    servidor <<- 1 
    tiempo_sig_evento[2] <<- reloj + rexp(1, 1/lambdaS)
  }
}

salidas <- function(){
  if(q_t == 0){
    servidor <<- 0
    tiempo_sig_evento[2] <<-  1e30
  } else {
    q_t <<- q_t - 1
    Di <<- reloj - lt_A[1]
    total_esperas <<- total_esperas + Di
    clientes_enespera <<- clientes_enespera + 1
    tiempo_sig_evento[2] <<- reloj + rexp(1, 1/lambdaS)
    for(i in 1:q_t) lt_A[i] <<- lt_A[i+1]
  }
}

reporte <- function(){
  print(paste("Promedio de espera en la fila:", round(total_esperas/clientes_enespera, 2), "minutos", sep=" "))
  print(paste("Número promedio de clientes esperando en la fila:",round(area_q/reloj, 2), sep = " "))
  print(paste("Utilización del servidor:",100*round(area_status_servidor/reloj, 2), "%", sep = " "))
  print(paste("El tiempo de simulación fue de:", round(reloj,2), "minutos", sep = " "))
  return(list(promedio.espera = total_esperas/clientes_enespera,
              longitud.promedio.fila = area_q/reloj,
              utilizacion = area_status_servidor/reloj,
              tiempo.simulacion = reloj))
}

actualiza_estadisticas <- function(){
  tiempo_desde_ultimo_evento <<- reloj - tiempo_ultimo_evento
  tiempo_ultimo_evento <<-  reloj
  area_q <<- area_q + q_t * tiempo_desde_ultimo_evento
  area_status_servidor <<- area_status_servidor + servidor * tiempo_desde_ultimo_evento
}

mm1()
