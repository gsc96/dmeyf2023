# Ensemble final florgiunegbagging + mixfedeveroezemiguel + joseleoisunbaltrue

require("data.table")
require("lightgbm")

PARAM <- list()
PARAM$experimento <- "ensemble_4_modelos"

# Leer los tres archivos
archivo1 <- read.table("~/buckets/b1/exp/intento-lightgbm-mix-1/prediccion.txt", header = TRUE)
archivo2 <- read.table("~/buckets/b1/exp/lightgbmnegbag/prediccion.txt", header = TRUE)
archivo3 <- read.table("~/buckets/b1/exp/joseleoisunbaltrue_mode_corregido/prediccion.txt", header = TRUE)
archivo4 <- read.table("~/buckets/b1/exp/KA8240/prediccion.txt", header = TRUE)


if (!all(names(archivo1)[1:2] == names(archivo2)[1:2] & names(archivo2)[1:2] == names(archivo3)[1:2] & names(archivo3)[1:2] == names(archivo4)[1:2])) {
  stop("Las dos primeras columnas no son iguales en los tres archivos.")
}

# Crear un nuevo dataframe con las dos primeras columnas iguales y la tercera columna como promedio
predicciones <- data.frame(
  archivo1[, 1:2],
  Promedio = rowMeans(cbind(archivo1[, 3], archivo2[, 3], archivo3[, 3], archivo4[, 3]), na.rm = TRUE)
)

# Guardar el nuevo dataframe en un nuevo archivo
write.table(predicciones, "nuevo_archivo.txt", sep = "\t", quote = FALSE, row.names = FALSE)

#------------------------------------------GENERO ENTREGA A KAGGLE------------------------------------------------#
# Ordeno por probabilidad descendente
setorder(predicciones, -Promedio)
setDT(predicciones)
setwd("~/buckets/b1/")
# Creo carpeta donde guardar los experimentos en caso de que no exista
dir.create("./exp/", showWarnings = FALSE)

# Creo carpeta donde guardar este experimento en caso de que no exista
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory de este experimento
setwd(paste0("./exp/", PARAM$experimento, "/"))


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