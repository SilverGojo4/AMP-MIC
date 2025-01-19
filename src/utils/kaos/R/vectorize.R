#'Vectorizes the FCGR
#'
#'@name vectorize
#'@param data  CGR object from cgr.R
#'
#'@details This function returns the FCGR as vector.
#'
#'@return Vector with FCGR-encoding
#'
#'@export
#'
#'
#'


vectorize = function (data) {
  as.vector(data$matrix)
}

