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
#INDICAR METODO DE IMPUTACION
metodo_imputacion <- "mode_corregido"

PARAM$experimento <- paste("ultimo_intento_semillerio_joseleounbaltrue",metodo_imputacion,sep="_")

PARAM$input$dataset <- "competencia_03_julia.csv"

# meses donde se entrena el modelo
PARAM$input$training <- c(202010,202011,202012, 202101, 202102, 202103,202104,202105,202106,202107)
PARAM$input$future <- c(202109) # meses donde se aplica el modelo

# Parámetro variable (esto genera semillas con valor entre 15k y 80k, puede ajustar a preferencia)
cantidad_semillas = 20 # Cuántas semillas desea ensamblar?
semillas <- as.integer(seq(15000, 80000, length.out = cantidad_semillas))

# hiperparametros intencionalmente NO optimos
PARAM$finalmodel$optim$num_iterations <- 2635
PARAM$finalmodel$optim$learning_rate <- 0.0311394911683338
PARAM$finalmodel$optim$feature_fraction <- 0.998656962501431
PARAM$finalmodel$optim$min_data_in_leaf <- 12638
PARAM$finalmodel$optim$num_leaves <- 217

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
  neg_bagging_fraction = 1.0, # 0.0 < neg_bagging_fraction <= 1.0
  is_unbalance = TRUE, #
  scale_pos_weight = 1.0, # scale_pos_weight > 0.0
  
  drop_rate = 0.1, # 0.0 < neg_bagging_fraction <= 1.0
  max_drop = 50, # <=0 median no limit
  skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0
  
  extra_trees = TRUE # Magic Sauce
  
)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)

# Feature Engineering Historico  ----------------------------------------------
campos_buenos <- setdiff(colnames(dataset), c('foto_mes','numero_de_cliente',"clase_ternaria", "clase01"))

# Catastrophe Analysis  -------------------------------------------------------

# zero ratio imputo nulos

dataset[foto_mes %in% c(202010, 202102), mcajeros_propios_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), ctarjeta_visa_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), mtarjeta_visa_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), ctarjeta_master_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), mtarjeta_master_descuentos := NA]
dataset[foto_mes == 202105, ccajas_depositos := NA]
#------------------------------------------------------------------------------
#Imputación de nulos

taining_subset <- dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105), ..campos_buenos]

# small_dataset <- dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105)]

# Defino formula para la moda
mode <- function(x, na.rm = FALSE) {
  
  if(na.rm){ #if na.rm is TRUE, remove NA values from input x
    x = x[!is.na(x)]
  }
  
  val <- unique(x)
  return(val[which.max(tabulate(match(x, val)))])
}

mode_training <- lapply(taining_subset, mode, na.rm = TRUE)

# Chequeo nulos
sum(is.na(dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105, 202107)]))

# Convierto integer a numeric
for (i in colnames(dataset)) {
  if (class(dataset[[i]]) == "integer") {
    dataset[[i]] <- as.numeric(dataset[[i]])
  }
}


# Reemplazo por la moda de training, en todos los datasets! Solo con esto andaría. Siguen habiendo problemas de truncamiento
for (col in names(taining_subset)) {
  dataset[is.na(dataset[[col]]), (col) := mode_training[[col]]]
}

# Chequeo que no hay nulos
sum(is.na(dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105, 202107)]))

#------------------------------------------------------------------------------
# LAGS 1, 3 y 6 ################################################################

#calculate lags 1,3 and 6 in campos_buenos
dataset[, paste0(campos_buenos, "_lag1") := lapply(.SD, shift, 1L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag3") := lapply(.SD, shift, 3L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag6") := lapply(.SD, shift, 6L, type = "lag"), .SDcols = campos_buenos]

################################################################################

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

# genero la tabla de entrega
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
tb_entrega[, proba_ensemble := predicciones]

# grabo las probabilidad del modelo
fwrite(tb_entrega,
       file = "prediccion.txt",
       sep = "\t"
)

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