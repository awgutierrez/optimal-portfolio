using CSV
using DataFrames
using StatsBase 
using YFinance
using ShiftedArrays

using JuMP
using HiGHS
using Plots

#Loading list of tickers of S&P 500 index
Tickers = DataFrame(CSV.File("SP500tickers.csv"))

#Selecting some tickers randomly from all the 
#s&p500 tickers
n_init = 20; 

sTickers = sample(axes(Tickers, 1), n_init; 
        replace = false, ordered = true);

Assets = Tickers[sTickers,1];

#Getting daily stock prices from Yahoo! Finance
data = get_prices.(Assets,
        startdt="2023-01-8",
        enddt="2024-05-30",
        throw_error=true);
		
data = data |> DataFrame

data[!,["ticker","open","close","adjclose"]]

#Checking that the necessary data is complete
#Some assets may have missing values in adjusted 
#closing price

# adding a column length(adjusted closing price) 
data[!,:"length_adjclose"] = [
    length(data[!,"adjclose"][i]) for i in 1:n_init];

# showing number of assets with incomplete data
N_aux = maximum(data[!,:"length_adjclose"]);
count(data[!,"length_adjclose"] .< N_aux)

#Setting the data
#selecting assets with full historical data
data = filter(:"length_adjclose" 
        => l -> l == N_aux, data); 

# number of assets in our portfolio
n = size(data,1);

# number of trading days
N = N_aux - 1; 

data[!,["ticker","timestamp","adjclose"]]

#Calculating daily returns
r = [zeros(N) for _ in 1:n]; aux1 = data[!,"adjclose"][:];
for i in 1:n
    aux2 = ShiftedArray(aux1[i],1,default=NaN) 
    r_aux = log.(aux1[i]./aux2)
    r[i] = r_aux[2:N_aux]
end
R = transpose(reduce(hcat,r));

#DataFrame(Matrix(R),:auto)

#Assumption : daily returns are equally likely scenarios
p = (1/N)*ones(N);

#Mean-semideviation model
m = Model(HiGHS.Optimizer); set_silent(m);
lambda = 0.8;

@variable(m, z[i=1:n] >= 0.0)     
@variable(m, s[j=1:N] >= 0.0) 
@expression(m,auz[k in 1:N],
    sum(p[j]*R[i,j]*z[i] for j in 1:N, i in 1:n) 
    - sum(R[i,k]*z[i] for i in 1:n))

@constraint(m, C[k in 1:N], s[k] >= auz[k])  
@constraint(m, sum(z) == 1) 
@objective(m, Min, -sum(z[i]*R[i,j]*p[j] for j in 1:N, i in 1:n) 
    + lambda*sum(p[j]*s[j] for j in 1:N)) 

optimize!(m);

#Optimal asset allocation
z_opt = value.(z); data[!,:"opt_alloc"] = z_opt;

data_to_show = filter(:"opt_alloc" => a -> a != 0.0, data);
data_to_show[!,[:"ticker",:"opt_alloc"]]

#Plot results
s=string(lambda)

ac = bar(data_to_show[:,"ticker"],
    data_to_show[:,"opt_alloc"].*100.00, 
    xrotation=60,xticks = :all,yticks = 0:5:100,
    title ="optimal asset allocation \n lambda = $s",
    label = false,
    xlabel = "S&P500 constituents",
    ylabel = "asset allocation (%)");
	
#Risk-adjusted measure
# Lagrange multipliers associated to the constraint in the model
u_opt = dual.(C); 

# risk-adjusted probabilities
q = (1-sum(u_opt))*p + u_opt 


opt_return = [sum(R[:,i].*z_opt) for i in 1:N];


sorted_adj = sortslices(hcat(opt_return,q), 
    dims = 1, by=col->(col[1], col[2]))

str = string(lambda)

plot_img = plot(
    sorted_adj[:,1], [(1:N)./N, cumsum(sorted_adj[:,2])], 
    title = "cdf portfolio returns \n lambda = $str", 
    linewidth=1, msw = 0, ms = 1.0,
    xlabel = "returns", yticks = 0:0.1:1.0,
    label = ["using original measure p" "using risk-adjusted measure q"]);
	
plot_img

fig_name = "cdf_returns-lambda=$str.pdf";
plot_lambda = plot(ac,plot_img,layout = (1,2));
savefig(plot_lambda,fig_name);

objective_value(m)