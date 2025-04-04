#'Produces a new matrix with new resolution
#'
#'@name cgr.res
#'@param data  CGR object
#'@param res new resolution
#'
#'@details This function adjust the resolution of the FCGR of a CGR object.
#'
#@return CGR object as list of:
#'\itemize{
#'     \item matrix: frequency matrix with new given resolution
#'     \item x: x-coordinates for the CGR
#'     \item y: y-coordinates for the CGR
#'     \item sf: applied scaling factor for the CGR
#'     \item res: applied resolution to calculate the FCGR
#'     \item base.seq: chars or letters to build the edges of the CGR
#'}
#'
#'@export
#'
#'
#'


cgr.res = function (data, res) {
  r = 1
  A = matrix(data = 0, ncol = res, nrow = res)
  for (i in 1:length(data$x)) {
    x.matrix = ceiling((data$x[i]+r ) * res/(2*r))
    y.matrix = ceiling((data$y[i]+r ) * res/(2*r))
    A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + 1

  }

  #return matrix, coordinates, kissing number, resolution
  return(list(matrix = A,
              x = data$x,
              y = data$y,
              scaling_factor = data$sf,
              resolution = res,
              base = data$base))
}
