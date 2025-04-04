


distr.pts = function(n,
                     r,
                     plot = F){

  #get coordinates for a regular polygon
  x = vector("double", n)
  y = vector("double", n)
  for (i in 1:n){
    x[i] = r*sinpi((2*i+1)/n)
    y[i] = r*cospi((2*i+1)/n)
  }

  #generates a plot if required
  if (plot) {plot(x, y, pch = 20)}

  #return coordinates
  return(xy.coords(x, y))
}
