#'Chaos Game Representation Object
#'
#'@param data Input as list/vector of characters from fasta file or similar
#'@param seq.base By default the included unique elements in data will be used
#'    in alphabetical order. It is also possible to define the alphabet
#'    explicitly.\cr
#'    Predefined alphabets can be used as well:\cr
#'\itemize{
#'    \item "digits": numbers from 0 to 9
#'    \item "AMINO": alphabetical order of the amino acids in capital letters
#'    \item "amino": alphabetical order of the amino acids in lowercase letters
#'    \item "DNA": the four bases of DNA ("A","G","T","C") in capital letters
#'    \item "dna": the four bases of DNA ("a","g","t","c") in lowercase letters
#'    \item "LETTERS": The alphabetical order of capital letters from A to Z
#'    \item "letters": The alphabetical order of lowercase Letters from a to z
#'}
#'@param sf By default, the scaling factor for fractal polygons is used; the
#'    scaling factor can also explicitly set to values between 0 and 1.
#'@param res resolution of the frequency matrix
#'
#'@details This function produces a chaos game representation (CGR)
#'    object from a sequence (data)
#'
#'@return CGR object as list of:
#'\itemize{
#'     \item matrix: frequency matrix with given resolution
#'     \item x: x-coordinates for the CGR
#'     \item y: y-coordinates for the CGR
#'     \item sf: applied scaling factor for the CGR
#'     \item res: applied resolution to calculate the FCGR
#'     \item base.seq: chars or letters to build the edges of the CGR
#'}
#'
#'@export
#'
#'@examples
#'###HIV data
#'data("HIV")
#'
#'### encoding the sequence
#'HIV.cgr = cgr(HIV, res = 100)
#'
#'###plot the sequence
#'cgr.plot(HIV.cgr, mode = "points")
#'
#'###plot the FCGR
#'cgr.plot(HIV.cgr, mode = "matrix")
#'
#'###change the resolution of matrix from 100x100 to 200x200
#'cgr.res(HIV.cgr, 200)
#'
#'### get the FCGR encoded vector
#'vectorize(HIV.cgr)
#'
#'


cgr = function(data,
               seq.base = row.names(table(data)),
               sf = F,
               res = 100) {

  r = 1
  if(is.character(seq.base)&&length(seq.base)==1){
    if(seq.base == "digits"){
      seq.base =c(0:9)
    }
    else if(seq.base == "AMINO"){
      seq.base=c("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R",
                 "S","T","V","W","Y")
    }
    else if(seq.base == "amino"){
      seq.base=c("a","c","d","e","f","g","h","i","k","l","m","n","p","q","r",
                 "s","t","v","w","y")
    }

    else if (seq.base == "DNA"){
      seq.base= c("A","G","T","C")
    }

    else if (seq.base == "dna"){
      seq.base= c("a","g","t","c")
    }
    else if (seq.base == "LETTERS"){
      seq.base=LETTERS
    }

    else if (seq.base == "letters"){
      seq.base=letters
    }

  }

  #check for input errors
  stopifnot(
    length(seq.base) >= length(table(data)),
    all(row.names(table(data)) %in% seq.base),
    sf <= 1,
    sf >= 0,
    res >= 1)




  #get the number of bases
  base.num = length(seq.base)

  if(base.num==4){
    x=c(1,-1,-1,1)
    y=c(-1,-1,1,1)
    base.coord = xy.coords(x, y)
  }
  #calculate corner coordinates for the base
  else{

    base.coord = distr.pts(base.num, r)
  }

  #calculate the "scaling factor" depending on the number of bases.
  #the scaling factor is the ratio of the radius of the circle that
  #passes through the centers of the touching subpolygons to the
  #radius of a circle that circumscribes the parent polygon.
  if (!sf) {sf =  1- (sinpi(1/base.num)/ (sinpi(1 / base.num) + sinpi (
    1/base.num + 2 * (floor (base.num/4) /base.num))))}

  #data frame for easy access
  base = data.frame(x = base.coord$x,
                    y = base.coord$y,
                    row.names = seq.base)

  #get the length of data
  data.length = length(data)

  #cgr algorithm:
  #start at point (0,0)
  #1. check next character
  #2. go a fraction of the way to the corresponding base, according to the
  #scaling factor
  #3. save coordinates of the point
  #repeat
  x = vector("double", data.length)
  y = vector("double", data.length)
  A = matrix(data = 0, ncol = res, nrow = res)
  pt = vector("double", 2)
  for (i in 1:data.length) {
    pt = pt + (unlist(base[data[i],]) - pt) * sf
    x[i] = pt[1]
    y[i] = pt[2]
    x.matrix = ceiling((x[i]+r ) * res/(2*r))
    y.matrix = ceiling((y[i]+r ) * res/(2*r))
    A[x.matrix, y.matrix] = A[x.matrix, y.matrix] + 1
  }

  #return matrix, coordinates, scaling factor, resolution
  return(list(matrix = A,
              x = x,
              y = y,
              scaling_factor = sf,
              resolution = res,
              base = base))
}
