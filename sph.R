x <- seq(1,100)/100

sph <- function(a,b){
	y <- c()
	for (i in 1:length(x)){
		if (x[i]<a){
			y <- append(y,min(b+1.5*(x[i]/a)-.5*(x[i]/a)^3,1))
		}
		else {
			y <- append(y,1)
		}
	}
	return(y)
}

plot(x,sph(.3,.5),type="l")
for (a in seq(40,90,10)){
	lines(x,sph(a/100,.5),type="l")
}
