gof.dpmalt <- function(y, alpha, nmc = 1e4, nsub, omega1, omega2, scale.data = FALSE, to.store = FALSE, df.imp=NULL, scale.inflation=NULL, scale.variance=FALSE, base="mvbeta", verbose=TRUE) {
    
    #-- data 'y' should be a n x p matrix
    
    y <- as.matrix(y)
    n <- nrow(as.matrix(y))
    p <- ncol(as.matrix(y))
    
    pp <- (p - 1) / 2 + 1
    if(base == "mvbeta"){
        alpha_factor <- alpha^pp
        if(missing(omega1)) omega1 <- pp + 1 / alpha_factor
        if(missing(omega2)) omega2 <- pp + 1 * alpha_factor
    } else {
        if(missing(omega1)) omega1 <- 1
        if(missing(omega2)) omega2 <- p*alpha
    }
    
    if(missing(nsub)) nsub <- min(100, p*(p+1))
    
    #------ scale y to make it mean zero and variance identity--------#
    
    if(scale.data){
        mean.y <- apply(y, 2, mean)
        var.y <- var(y)
        sqrt.var.y <- chol(var.y)
        y <- t(apply(y, 1, function(z) return(backsolve(sqrt.var.y, z - mean.y, transpose = TRUE))))
    }
    dim(y) <- c(n, p)
    
    mean.y <- apply(y, 2, mean)
    var.y <- var(y)
    sqrt.var.y <- chol(var.y)
    inv.sqrt.var.y <- backsolve(sqrt.var.y, diag(1, p))

    inv.sqrt.scale <- diag(1,p)
    if(any(as.logical(scale.variance))){
        if(length(scale.variance) < p*p) scale.variance <- diag(scale.variance, p)
        infl.mat <- chol(scale.variance)
        scale.mat <- t(infl.mat) %*% var.y %*% infl.mat
        sqrt.scale <- chol(scale.mat)
        inv.sqrt.scale <- backsolve(sqrt.scale, diag(1,p))
    }
    
    if(is.null(scale.inflation)) scale.inflation <- sqrt(n)
    if(is.null(df.imp)) df.imp <- max(p, sqrt(n-1))
    
    out <- .C("dpm_gof_mv", y = as.double(y), mean.y = as.double(mean.y),
              inv.sqrt.var.y = as.double(inv.sqrt.var.y),
              inv.sqrt.scale = as.double(inv.sqrt.scale),
              dims = as.integer(c(n, p, nmc, to.store, nsub)),
              hpar = as.double(c(alpha, omega1, omega2, df.imp, scale.inflation)),
              lmarg.null = double(1), lwt.null = double(nmc), lwt.alt = double(nmc),
              sso = double(nmc), limpd = double(nmc),
              mu.store = double(1 + to.store * (nmc*p - 1)),
              sqrt.inv.sig.store = double(1 + to.store * (nmc*p*p - 1)),
              nclust.store = integer(nmc), mvbeta.base=as.integer(base=="mvbeta"))
    
    lwt <- out$lwt.alt
    out$lmarg.alt <- logmean(lwt)
    logBF <- out$lmarg.null - out$lmarg.alt
    wt <- exp(lwt - logsum(lwt))
    ess <- round(1/sum(wt^2))
    out$ESS <- ess
    out$logBF <- logBF
    

    if(verbose){
        cat("alpha = ", round(alpha, 3),
        "\tomegas = ", round(c(omega1, omega2), 2),
        "\tnsub = ", nsub,
        "\tESS = ", ess,
        "\n")
    }
    class(out) <- "gaussdpmalt"
    return(out)
}

plot.gaussdpmalt <- function(object, ...){
    obj <- object
    if(length(obj$mu.store) == 1) stop("Sampled draws were not stored while running importance sampler. Run again with 'to.store' set to TRUE")
    wt.null <- exp(obj$lwt.null - logsum(obj$lwt.null))
    wt.alt <- exp(obj$lwt.alt - logsum(obj$lwt.alt))
    p <- obj$dims[2]
    nmc <- obj$dims[3]
    mu <- matrix(obj$mu.store, p, nmc)
    sqrt.inv.sig <- matrix(obj$sqrt.inv.sig.store, p^2, nmc)
    for(j in 1:p){
        plot(density(mu[j,], weights = wt.null), col = 2, ty = "l", bty = "n", ann = FALSE)
        lines(density(mu[j,]))
        lines(density(mu[j,], weights = wt.alt), col = 3)
        #lines(density(mu[j,], weights = wt.ratio), col = 4)
        title(xlab = bquote(mu[.(j)]), ylab = "Density")
        legend("topright", c("Imp", "Null", "Alt-1"), lty = 1, col = 1:3, bty = "n")
    }
    for(j in 1:p){
        for(i in j:p){
            k <- p*(j-1) + i
            plot(density(sqrt.inv.sig[k,], weights = wt.null), col = 2, ty = "l", bty = "n", ann = FALSE)
            lines(density(sqrt.inv.sig[k,]))
            lines(density(sqrt.inv.sig[k,], weights = wt.alt), col = 3)
            #lines(density(sqrt.inv.sig[k,], weights = wt.ratio), col = 4)
            title(xlab = bquote(invsigma[.(i)][.(j)]), ylab = "Density")
            legend("topright", c("Imp", "Null", "Alt-1"), lty = 1, col = 1:3, bty = "n")
        }
    }
}

summary.gaussdpmalt <- function(object, print=TRUE, ...){
    obj <- object
    if(length(obj$mu.store) == 1) stop("Sampled draws were not stored while running importance sampler. Run again with 'to.store' set to TRUE")
    wt.null <- exp(obj$lwt.null - logsum(obj$lwt.null))
    wt.alt <- exp(obj$lwt.alt - logsum(obj$lwt.alt))

    p <- obj$dims[2]
    nmc <- obj$dims[3]
    mu <- matrix(obj$mu.store, p, nmc)
    sqrt.inv.sig <- matrix(obj$sqrt.inv.sig.store, p^2, nmc)
    
    get.sigma <- function(w) return(crossprod(solve(matrix(w,p,p,byrow=TRUE))))
    sigma <- matrix(apply(sqrt.inv.sig, 2, get.sigma), p^2, nmc)
    
    mu.est.null <- apply(mu, 1, weighted.mean, w=wt.null)
    sigma.est.null <- matrix(apply(sigma, 1, weighted.mean, w=wt.null),p,p)

    mu.est.alt <- apply(mu, 1, weighted.mean, w=wt.alt)
    sigma.est.alt <- matrix(apply(sigma, 1, weighted.mean, w=wt.alt),p,p)
    
    lo <- list('log-Bayes Factor of NULL vs ALT' = obj$logBF,
               'ESS' = obj$ESS,
               'NULL posterior mu' = mu.est.null,
               'NULL posterior Sigma' = sigma.est.null,
               'ALT posterior mu' = mu.est.alt,
               'ALT posterior Sigma' = sigma.est.alt)
    if(print) print(lo, ...)
    invisible(lo)
}

logsum <- function(lx) return(max(lx) + log(sum(exp(lx - max(lx)))))
logmean <- function(lx) return(logsum(lx) - log(length(lx)))
essFn <- function(lwt){
    wt <- exp(lwt - logsum(lwt))
    return(round(1/sum(wt^2)))
}

