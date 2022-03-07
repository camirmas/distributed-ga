function ackley(x; a=20, b=0.2, c=2π)
    sleep(0.01)
    d = length(x)
    return -a*exp(-b*sqrt(sum(x.^2)/d)) -
              exp(sum(cos.(c*xi) for xi in x)/d) + a + exp(1)
end
