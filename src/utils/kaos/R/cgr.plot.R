#'Plot from a CGR object
#'
#'@name cgr.plot
#'@param data CGR object
#'@param mode character string
#'\itemize{
#'     \item "points": CGR plot
#'     \item "matrix": FCGR plot
#'}
#'@param corners if true, the corners are added as red dots
#'@param labels if true, the symbol associated with the corner is added
#'
#'@details This function plots the chaos game reprasentation as points or as
#'frequency matrix representation
#'
#'
#'@export
#'
#' @import ggplot2
#' @import reshape2
#' @importFrom grDevices xy.coords
#' @importFrom graphics plot.new plot.window points text




cgr.plot = function(data, mode, corners = F, labels = F) {
  if (mode == "matrix") {
    Var1<-Var2<-value<-x<-y<-NULL
    matrixplot = ggplot(melt(data$matrix), aes(x = Var1, y = Var2)) +

      geom_raster(aes(fill = value)) +

      theme_bw() +

      theme(panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line=element_blank(),
            axis.text.x=element_blank(),
            axis.text.y=element_blank(),
            axis.ticks=element_blank(),
            axis.title.x=element_blank(),
            axis.title.y=element_blank(),
            legend.position="none")+

      scale_fill_gradient(low = "white", high="black") +

      labs(x = "", y = "", title = "")
    if (corners){
      matrixplot = matrixplot + geom_point(data = data.frame(
        x = (data$base$x + 0.5)*data$resolution,y = (
          data$base$y + 0.5)*data$resolution),
        mapping = aes(x, y),
        colour = "red",
        shape = 20)
    }
    if (labels){
      matrixplot = matrixplot + geom_text(data = data.frame(x = (
        data$base$x*1.1 + 0.5)*data$resolution,y = (
          data$base$y*1.1 + 0.5)*data$resolution),
        mapping = aes(x, y),
        label = rownames(data$base))
    }
    matrixplot
  }
  else if (mode == "points") {
    r = 0.5
    plot.new()
    plot.window(xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1))
    points(data$y ~ data$x, pch = 46)
    if (corners){
      points(data$base$y ~ data$base$x, col = "red", pch = 20, cex = 0.7)
    }
    if (labels){
      text(x = data$base$x*1.1, y = data$base$y*1.1, labels = rownames(
        data$base))
    }
  }
}
