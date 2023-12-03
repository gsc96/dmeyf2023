# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM


# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")


# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
PARAM$experimento <- "intento-lightgbm-mix-1-SEMILLERIO"

PARAM$input$dataset <- "competencia_03_julia.csv"

# meses donde se entrena el modelo
PARAM$input$training <- c(201907, 201908, 201909, 201910, 201911, 201912, 
                          202001, 202002, 202003, 202004, 202005, 202006, 
                          202007, 202008, 202009, 202010, 202011, 202012, 
                          202101, 202102, 202103,202104,202105,202106)
PARAM$input$future <- c(202109) # meses donde se aplica el modelo

# Parámetro variable (esto genera semillas con valor entre 15k y 80k, puede ajustar a preferencia)
cantidad_semillas = 40 # Cuántas semillas desea ensamblar?
semillas <- as.integer(seq(15000, 80000, length.out = cantidad_semillas))

# hiperparametros intencionalmente NO optimos
PARAM$finalmodel$optim$num_iterations <- 481
PARAM$finalmodel$optim$learning_rate <-0.0724641881672392
PARAM$finalmodel$optim$feature_fraction <- 0.954567477393551
PARAM$finalmodel$optim$min_data_in_leaf <- 49993
PARAM$finalmodel$optim$num_leaves <- 661
PARAM$finalmodel$optim$scale_pos_weight <- 1


# Hiperparametros FIJOS de  lightgbm
PARAM$finalmodel$lgb_basicos <- list(
  boosting = "gbdt", # puede ir  dart  , ni pruebe random_forest
  objective = "binary",
  metric = "custom",
  first_metric_only = TRUE,
  boost_from_average = TRUE,
  feature_pre_filter = FALSE,
  force_row_wise = TRUE, # para reducir warnings
  verbosity = -100,
  max_depth = -1L, # -1 significa no limitar,  por ahora lo dejo fijo
  min_gain_to_split = 0.0, # min_gain_to_split >= 0.0
  min_sum_hessian_in_leaf = 0.001, #  min_sum_hessian_in_leaf >= 0.0
  lambda_l1 = 0.0, # lambda_l1 >= 0.0
  lambda_l2 = 0.0, # lambda_l2 >= 0.0
  max_bin = 31L, # lo debo dejar fijo, no participa de la BO
  
  bagging_fraction = 1.0, # 0.0 < bagging_fraction <= 1.0
  pos_bagging_fraction = 1.0, # 0.0 < pos_bagging_fraction <= 1.0
  neg_bagging_fraction = 0.170533043714281, # 0.0 < neg_bagging_fraction <= 1.0
  is_unbalance = FALSE, #
  
  drop_rate = 0.1, # 0.0 < neg_bagging_fraction <= 1.0
  max_drop = 50, # <=0 means no limit
  skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0
  
  extra_trees = TRUE, # Magic Sauce
  zero_as_missing = TRUE,
  
  seed = PARAM$finalmodel$semilla
)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)


# Catastrophe Analysis  -------------------------------------------------------
# deben ir cosas de este estilo
#   dataset[foto_mes == 202006, active_quarter := NA]

# Data Drifting
# por ahora, no hago nada


# Feature Engineering Historico  ----------------------------------------------
#   aqui deben calcularse los  lags y  lag_delta
#   Sin lags no hay paraiso ! corta la bocha
#   https://rdrr.io/cran/data.table/man/shift.html
# FI: Coloco NA a todos los registos en 0
zero_ratio <- list(
  list(mes = 202006, campo = 
         c("active_quarter", "internet", "mrentabilidad", "mrentabilidad_annual", 
           "mcomisiones", "mactivos_margen", "mpasivos_margen", "mcuentas_saldo", 
           "ctarjeta_debito_transacciones","mautoservicio", "ctarjeta_visa_transacciones", 
           "mtarjeta_visa_consumo","ctarjeta_master_transacciones", "mtarjeta_master_consumo",
           "ccomisiones_otras", "mcomisiones_otras","cextraccion_autoservicio","mextraccion_autoservicio",
           "ccheques_depositados","mcheques_depositados","ccheques_emitidos","mcheques_emitidos",
           "ccheques_depositados_rechazados","mcheques_depositados_rechazados","ccheques_emitidos_rechazados",
           "mcheques_emitidos_rechazados","tcallcenter","ccallcenter_transacciones","thomebanking",
           "chomebanking_transacciones","ccajas_transacciones","ccajas_consultas","ccajas_depositos",
           "ccajas_extracciones","ccajas_otras","catm_trx","matm","catm_trx_other","matm_other",
           "tmobile_app","cmobile_app_trx")),
  list(mes = 201910, campo = 
         c("mrentabilidad", "mrentabilidad_annual","mcomisiones","mactivos_margen","mpasivos_margen",
           "ccomisiones_otras","mcomisiones_otras","chomebanking_transacciones")),
  list(mes = 201905, campo = 
         c("mrentabilidad", "mrentabilidad_annual", "mcomisiones","mactivos_margen","mpasivos_margen",
           "ccomisiones_otras","mcomisiones_otras")),
  list(mes = 201904, campo = 
         c("ctarjeta_visa_debitos_automaticos","mttarjeta_visa_debitos_automaticos"))
)

for (par in zero_ratio) {
  mes <- par$mes
  feature <- par$campo
  dataset[foto_mes == mes, (feature) := lapply(.SD, function(x) ifelse(x == 0, NA, x)), .SDcols = feature]
}

#______________________________________________________________
# FI: hago lag de los ultimos 6 meses de todas las features (menos numero cliente, foto mes y clase ternaria)

all_columns <- setdiff(
  colnames(dataset),
  c("numero_de_cliente", "foto_mes", "clase_ternaria")
)

setorder(dataset, numero_de_cliente, foto_mes)

periods <- c(1,3,6) # Seleccionar cantidad de periodos 

for (i in periods){
  lagcolumns <- paste("lag", all_columns,i, sep=".")
  dataset[, (lagcolumns):= shift(.SD, type = "lag", fill = NA, n=i), .SDcols = all_columns,  by =numero_de_cliente]
}

# Delta LAG de 1 y 2 periodos

for (vcol in all_columns){
  dataset[, paste("delta", vcol,1, sep=".") := get(vcol) - get(paste("lag", vcol,1, sep="."))]
}

for (vcol in all_columns){
  dataset[, paste("delta", vcol,3, sep=".") := get(vcol) - get(paste("lag", vcol,3, sep="."))]
}

for (vcol in all_columns){
  dataset[, paste("delta", vcol,6, sep=".") := get(vcol) - get(paste("lag", vcol,6, sep="."))]
}

#_______________________________________________
# Fin FE
#--------------------------------------


#--------------------------------------

# paso la clase a binaria que tome valores {0,1}  enteros
# set trabaja con la clase  POS = { BAJA+1, BAJA+2 }
# esta estrategia es MUY importante
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------

# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

#--------------------------------------


# establezco donde entreno
dataset[, train := 0L]
dataset[foto_mes %in% PARAM$input$training, train := 1L]

#--------------------------------------
# creo las carpetas donde van los resultados
# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))



#----------------------------------CONFIGURAR DATOS DE ENTRADA MODELO----------------------------------#
# Dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
  label = dataset[train == 1L, clase01]
)

#----------------------------------ITERACIÓN----------------------------------#

# Obtengo los datos a predecir
dapply <- dataset[foto_mes == PARAM$input$future]

# Selecciono columna con numero de cliente y foto mes en df para guardar las predicciones
predicciones <- dapply[, list(numero_de_cliente, foto_mes)]

cat("\n\nEmpieza la iteración, hora:", Sys.time(), "\n")

for (semilla in semillas) {
  #----------------------------------CONFIGURAR MODELO--------------------------------------------#
  # Utilizo los parámetros configurados al inicio para el modelo
  
  # genero el modelo
  param_completo <- c(PARAM$finalmodel$lgb_basicos,
                      PARAM$finalmodel$optim,
                      seed = semilla)
  
  modelo <- lgb.train(
    data = dtrain,
    param = param_completo,
  )
  
  #----------------------------------PERSISTIR IMPORTANCIA DE VARIABLES---------------------------------#
  # Este paso guarda la importancia de variables de cada modelo generado, puede descomentar si desea guardarlas)
  # Calculo la importancia de variables del modelo
  # tb_importancia <- as.data.table(lgb.importance(modelo))
  
  # Configuro nombre del archivo
  # archivo_importancia <- paste0("impo_", semilla, ".txt")
  
  # Guardo en el archivo 
  # fwrite(tb_importancia,
  # file = archivo_importancia,
  # sep = "\t"
  #)
  
  #----------------------------------PREDECIR SOBRE MES DE INTERÉS---------------------------------#
  # Aplico el modelo a los nuevos datos
  prediccion <- predict(
    modelo,
    data.matrix(dapply[, campos_buenos, with = FALSE])
  )
  
  # Agrego columna con las predicciones de cada semilla
  col_name <- paste0("semilla_", semilla)
  predicciones[, (col_name) := prediccion] 
  cat("\n\nSemilla número", semilla , "hora:", Sys.time(), "\n")
  
}

#-------------------------------PERSISTO SALIDA CON LAS PREDICCIONES DE CADA SEMILLA------------------------------#
# Guardo el archivo (con probas)
archivo_salida <- paste0(PARAM$experimento, "_predicciones_semillas.csv")
fwrite(predicciones, file = archivo_salida, sep = ",")

#-----------------------------------------------GENERO ENSEMBLE---------------------------------------------------#

# Calcular el promedio de las predicciones (probas) de los 100 modelos ejecutados (excluyo cols "numero_de_cliente" y "foto_mes")
predicciones$proba_ensemble <- rowMeans(predicciones[, .SD, .SDcols = -(1:2)])

cat("\n\nEnsemble generado, hora:", Sys.time(), "\n")

#------------------------------------------GENERO ENTREGA A KAGGLE------------------------------------------------#
# Ordeno por probabilidad descendente
setorder(predicciones, -proba_ensemble)


# Genero archivos variando la cantidad de estímulos
cortes <- seq(8000, 15000, by = 500)
for (envios in cortes) {
  predicciones[, Predicted := 0L]
  predicciones[1:envios, Predicted := 1L]
  
  fwrite(predicciones[, list(numero_de_cliente, Predicted)],
         file = paste0(PARAM$experimento, "_", envios, ".csv"),
         sep = ","
  )
}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado, hora:", Sys.time(),"\n")