# to compute the ion concentration in the water beam
# for high pressure ( in the beginning ) a langevin model is used
# for the low pressure part a thomson model is applied
# the models are described with a system of coupled differential equations
# which is numerically solved by the scipy.signal.odeint routine



import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt



##################################################
A2YN = 0 # set 0 or 1: includes or excludes clustering via alpha2
### parameter to represent clustering in coupled ODEs
r0rnu = 0.02 # taken from Bates1983, relevant for alpha2L
sizefactor = 4 # division factor for mean free path in alpha3T
### variables
T = 1000. # [K]
v = 6000. # [m/s]
#p0_bar = 4600. # [bar] # Charvat2006 # is now calculated from density
r0_waterJet = 10.*10**( -6 ) # [m]
h_waterJet = 33*10**( -6 ) # [m]
volumeFraction = 1. # .1 represents the outer 500nm of the water jet. chosen to fit model data to experimental data
simulationDuration = 100.*10**(-9)
molarAccuracy = .1
### parameters
#epsilonr = 1 # of the water # Fernandez1997 fig23 points to 8-12, but only for large p, therefor neglecte for now
polarizability_gas = 1.501 # [Angstrom**3] for water # http://cccbdb.nist.gov/ last checked at 26.01.2015
dipolmoment_NaCl = 28.36*10**-30 # [Cm]
mass_gas = 18. # [amu]
mass_positive_ion = 23. # [amu]
mass_negative_ion = 35.5 # [amu]
gas_molecule_size = 2.8*10**( -10 ) # [m] diameter
### constants
epsilon0 = 8.854 * 10**(-12) # [As/Vm]
e = 1.602 * 10**( -19 ) # [C]
kB = 1.381 * 10**( -23 ) # [J/K]
pi = math.pi
amu = 1.661 * 10**( -27 ) # [kg]
### derived variables
#epsilon = epsilonr * epsilon0 # [As/Vm]
# p0 = p0_bar*10**5 # [Pa] #old
p0 = 8.3 * T / 0.018 * 1000 # [Pa]
v_gas = v # [m/s]
v_pos = v # math.sqrt( 2./mass_positive_ion/amu * 3./2*kB*T ) # [m/s] 1473m/s bei T=2kK
v_neg = v # math.sqrt( 2./mass_negative_ion/amu * 3./2*kB*T ) # [m/s] 1185m/s bei T=2kK
V0 = pi*r0_waterJet**2*h_waterJet*volumeFraction # [m**3]
r0_sphere = ( 3./4/pi*V0 )**( 1./3. ) # the radius of a sphere with the same volume as the water jet at t0
red_pos_mass = mass_gas * mass_positive_ion / ( mass_gas + mass_positive_ion ) #[amu]
red_neg_mass = mass_gas * mass_negative_ion / ( mass_gas + mass_negative_ion ) #[amu]
red_ion_mass = mass_positive_ion * mass_negative_ion / ( mass_positive_ion + mass_negative_ion ) # [amu]
#d0_ionion = 1./( 4.*pi*epsilon ) * 2.*e**2./( 3.*kB*T ) # [m]
#d0_iondipol = math.sqrt( 1./( 4.*pi*epsilon ) * 2.*e*dipolmoment_NaCl/( 3.*kB*T ) ) # [m]
##################################################



def epsilon( t ):
    #return epsilon0
    epsilonr = 1+10 * ( p(t)/p0 )
    ###Uematso1980 - but this formulation is not valid (negative epsilonr) for p > 10kbar
    #Ts = T/298.15
    #ps = p( t ) / p0
    #epsilonr = 1 + 7.63/Ts*ps + (244./Ts - 141 + 27.8*Ts)*ps**2 + (-96.2/Ts + 41.8*Ts - 10.2*Ts**2)*ps**3 + (-45.2/Ts**2 + 84.6/Ts - 35.9)*ps**4
    return epsilon0*epsilonr



def d0ionion( t ):
    d0ii = 1./( 4.*pi*epsilon( t ) ) * 2.*e**2./( 3.*kB*T ) # [m]
    return d0ii



def d0iondipol( t ):
    d0id = math.sqrt( 1./( 4.*pi*epsilon( t ) ) * 2.*e*dipolmoment_NaCl/( 3.*kB*T ) ) # [m]
    return d0id



def l( t ): # [m]
# mean free path length, from wikipedia
    return kB*T / ( math.sqrt( 2. ) * pi * gas_molecule_size**2 * p( t ) )



def V( t ): #[m**3]
# volume of water plume
    r_t = r0_sphere + v_gas * t
    V_t = 4./3*pi* r_t**3
    return V_t



def p( t ): #[Pa]
# pressure in water plume
    p_t = p0 * V0 / V( t )
    return p_t



def mobility_positive( t ): #[m**2/( V*s )] 
# mobility of positive ions
    # for polarizability in [angstrom**3] and reduced mass in [amu]
    return 13.87 / math.sqrt( polarizability_gas * red_pos_mass ) * 10**5/p( t ) * 10**( -4 ) # *T/293. # from 2013 - why?
    # alternative: measurements from Tyndall in 1938 - comparable to the formula above
    #return 0.715 *10**( -4 ) * T / 293 * 1.01*10**5/p( t )



def mobility_negative( t ): #[m**2/( V*s )] 
# mobility of negative ions
    # for polarizability in [angstrom**3] and reduced mass in [amu]
    return 13.87 / math.sqrt( polarizability_gas * red_neg_mass ) * 10**5/p( t ) * 10**( -4 ) # *T/293. # from 2013 - why?



def f( t, d0, mfp ):
# factor used for the ion recombination rate in Thomson's model representing collision probability
    if mfp == "ideal":
        lambd = l( t )
    elif mfp == "sizefactor":
        lambd = l( t )/sizefactor
    else:
        print "wrong mean free path given: " + mfp
        exit(  )
    x_t = 2.*d0/lambd 
    w_x = 1. - 2./x_t**2 *( 1. - math.e**( -x_t )*( x_t+1 ) )
    f_x = 2.*w_x - ( w_x )**2
    return f_x



##################################################



def alpha_L( t, M ):
# ion recombination rate according Langevin model 
# valid in the high pressure regime
    return e/epsilon( t ) * ( mobility_positive( t ) + mobility_negative( t ) ) * ( 1. )#- ( M+1. )/3. ) # Harper1932



def alpha_T( t, d0v, mfp ):
# ion recombination rate according to Thomson model 
# valid in the low pressure regime
# taken from Dessiaterik2003
    if d0v == "ionion":
        d0 = d0ionion( t )
    elif d0v == "iondipol":
        d0 = d0iondipol( t )
    else:
        print "wrong d0v given: " + d0v
        exit(  )
    return pi * d0**2 * math.sqrt( v_pos**2 + v_neg**2 ) * f( t, d0, mfp )
    #C5_S8
    #g12 = math.sqrt( 8.*kB*T/pi/red_ion_mass/amu )
    #return g12 * 4.*pi*d0**3/3. * ( 2./l( t ) ) 



def alpha( t, M ):
# combination of the Langevin and Thomson model for wide range pressures
    # to take a continous transition from Langevin's to Thomson's model
    alpha = np.minimum( [alpha_L( t, M )], [alpha_T( t, "ionion", "ideal" )] ) 
    return alpha[0]



def alpha_arr( t, M ):
# in order to better work with plot command, just used for plotting
# nearly the same as alpha(  ), but takes different input format
    alpha = np.array( [] )
    for i in t:
        alpha = np.append( alpha, np.minimum( [alpha_L( i, M )], [alpha_T( i, "ionion", "ideal" )] ) )
    return alpha

def alpha2_arr( t, M ):
# in order to better work with plot command, just used for plotting, as above
    alpha = np.array( [] )
    for i in t:
        alpha = np.append( alpha, alpha2( i, M ) )
    return alpha

def alpha3c_arr( t, M ):
# in order to better work with plot command, just used for plotting, as above
    alpha = np.array( [] )
    for i in t:
        alpha = np.append( alpha, alpha3c( i, M ) )
    return alpha



#################################################
### extension of Thomsons and Langevins models 
### as presented in Dessiaterik2003a 
### with a system of coupled ODEs to include more clustering



def red_mass( m_ion ):
# reduced mass of an ion with water molecule
    return 18. * m_ion / ( 18. + m_ion )

def compare_alpha_L( m_pos, m_neg ):
# comparison factor for alpha_L of two new ions compared to Na+ and Cl-
# respecting the mass difference only
# used to represent mass change in alpha3L
    neu = math.sqrt( 1./red_mass( m_pos ) ) + math.sqrt( 1./red_mass( m_neg ) )
    Na_Cl = math.sqrt( 1./red_mass( 23. ) ) + math.sqrt( 1./red_mass( 35.5 ) )
    return neu / Na_Cl

def compare_alpha_T( m_pos, m_neg ): 
### in 2016 I can't remember what this was for in 2013, so it's' disabled and returns 1
# comparison factor for alpha_T of two new ions compared to Na+ and Cl-
# respecting the mass difference only
    return 1 #s.o. disabled
    neu = math.sqrt( 1/m_pos + 1/m_neg )
    Na_Cl = math.sqrt( 1/23. + 1/35.5 )
    return neu / Na_Cl

def alpha2( t,M ):
# recombination rate of NaCl and Na+
    alpha2 = np.minimum( [alpha_L( t, M )*r0rnu], [alpha_T( t, "iondipol", "ideal" )] ) 
    return alpha2[0]*A2YN

def alpha3a( t,M ):
# recombination rate of Na+ and NaClCl-
# respecting the mass difference only from NaClCl- to Cl-
    caL = compare_alpha_L( 23., 94. )
    caT = compare_alpha_T( 23., 94. )
    alpha = np.minimum( [alpha_L( t, M )*caL], [alpha_T( t, "ionion", "sizefactor" )*caT] ) 
    return alpha[0]

def alpha3b( t,M ):
# recombination rate of Cl- and NaClNa+
# respecting the mass difference only from NaClNa+ to Na+
    caL = compare_alpha_L( 81., 35.5 )
    caT = compare_alpha_T( 81., 35.5 )
    alpha = np.minimum( [alpha_L( t, M )*caL], [alpha_T( t, "ionion", "sizefactor" )*caT] ) 
    return alpha[0]

def alpha3c( t,M ):
# recombination rate of NaClNa+ and NaClCl-
# respecting the mass difference only from NaClCl- to Cl- and NaClNa+ to Na+
    caL = compare_alpha_L( 81., 94. )
    caT = compare_alpha_T( 81., 94. )
    alpha = np.minimum( [alpha_L( t, M )*caL], [alpha_T( t,"ionion", "sizefactor" )*caT] ) 
    return alpha[0]

#################################################



def C_dot( C, t, M, tmp ):
# system of coupled differential equations to describe ion concentrations
# using scipy.integrate.odeint
# tmp does not get used, but without it errors are produced
    return np.array([ -alpha( t, M )*C[0]**2 -alpha3a( t,M )*C[2]*C[0], #!! in 2016 verwendete Zeile, der jedoch ein Term fehlt!!
    #return np.array([ -(alpha( t, M )*C[0]**2) -(alpha2( t,M )*C[1]*C[0]) -(alpha3a( t,M )*C[2]*C[0]), # korrigierte Zeile in 2017
        +alpha( t,M )*C[0]**2 - alpha2( t,M )*C[1]*C[0]*2,
        +alpha2( t,M )*C[1]*C[0] - alpha3b( t,M )*C[2]*C[0] - alpha3c( t,M )*C[2]**2 ])


    
#################################################



### ODE settings
t = np.arange( 0, simulationDuration, 100.*10**( -12 ) ) # [s] # times for numerical integration
molar_conc = np.arange( -7, -.4, molarAccuracy)[::-1] # initial molar ion concentrations



plt.figure( "alpha", figsize=(12,9) )
plt.xlabel( "time [$s$]", size=35 )
plt.ylabel( r"recombination coefficient [$m^3/s$]", size=35 )
plt.xscale( "log" )
plt.yscale( "log" )
plt.yticks( fontsize= 25 )
plt.xticks( fontsize= 25 )
if A2YN == 0:
    plt.plot( t, alpha_L( t, 0 ), color='b', lw=2 )
    plt.plot( t, alpha_T( t, "ionion", "ideal" ), color='r', lw=2 )
    plt.plot( t, alpha_arr( t, 0 ), color='g', lw=2 )
    plt.savefig( "CtcODE_alpha.png", bbox_inches="tight" )
    plt.savefig( "CtcODE_alpha.pdf", bbox_inches="tight", dpi=600 )
    np.savetxt( "CtcODE_alpha.txt", np.concatenate( ( t.reshape( t.size,1 ), alpha_L( t,0 ).reshape( t.size,1 ), alpha_T( t, "ionion", "ideal" ).reshape( t.size,1 ), alpha_arr( t,0 ).reshape( t.size,1 ) ), axis=1  ) )
elif A2YN == 1:
    plt.plot( t, alpha_arr( t, 0 ), color='g', lw=2 )
    plt.plot( t, alpha2_arr( t, 0 ), color='b', lw=2 )
    plt.plot( t, alpha3c_arr( t, 0 ), color='r', lw=2 )
    plt.savefig( "CtcODE_alpha123.png", bbox_inches="tight" )
    plt.savefig( "CtcODE_alpha123.pdf", bbox_inches="tight", dpi=600 )
    np.savetxt( "CtcODE_alpha123.txt", np.concatenate( ( t.reshape( t.size,1 ), alpha_L( t,0 ).reshape( t.size,1 ), alpha_T( t, "ionion", "ideal" ).reshape( t.size,1 ), alpha_arr( t,0 ).reshape( t.size,1 ) ), axis=1  ) )

plt.figure( "pressure", figsize=(12,9 ) )
plt.xlabel( "time [s]", size=35 )
plt.ylabel( "pressure [bar]", size=35 )
plt.xscale( "log" )
plt.yscale( "log" )
plt.yticks( fontsize= 25 )
plt.xticks( fontsize= 25 )
plt.plot( t, p( t )/10**5, color='g', lw=2 )
plt.savefig( "CtcODE_pressure.png", bbox_inches="tight" )
plt.savefig( "CtcODE_pressure.pdf", bbox_inches="tight", dpi=600 )
np.savetxt( "CtcODE_pressure.txt", np.concatenate( ( t.reshape( t.size,1 ), p( t ).reshape( t.size,1 ) ), axis=1 ) )

plt.figure( "ion count over time", figsize=(12,9) )
plt.xlabel( "time [s]", size=35 )
plt.ylabel( "concentration [mol / L]", size=35 )
plt.xscale( "log" )
plt.yscale( "log" )
plt.yticks( fontsize= 25 )
plt.xticks( fontsize= 25 )
final_Na_count = np.array( [] )
final_NaClNa_count = np.array( [] )
for M in molar_conc:
    # C0 contains the initial concentrations of Na+, NaCl, (NaCl)Na+
    C0 = np.array([ 6.*10**( 23. + M ), 0., 0. ])
    y = odeint( C_dot , C0, t, ( M, 1 ) ) # actual numerical integration
    y = np.concatenate( ( np.array( [M,0,0] ).reshape( 1,3 ), y[:-1,:] ) ) # to have M as first value
    y = y/6/10**23 # [mol/L]
    plt.plot( t, y[:,0], label= r'C$_{0}$ = $10^{' + str( M ) +'}$ M' ) # to print counts of Na
    #plt.plot( t, y[:,2], label= r'C$_{0}$ = $10^{' + str( M ) +'}$ M' ) # to print counts of NaClNa
    #print y[-1,:]
    final_Na_count = np.append( final_Na_count, y[-1,0] )
    final_NaClNa_count = np.append( final_NaClNa_count, y[-1,2] )
final_ion_count = final_Na_count + final_NaClNa_count
plt.legend( loc=3, prop={'size':25})
plt.savefig( "CtcODE_counts.png", bbox_inches="tight" )
plt.savefig( "CtcODE_counts.pdf", bbox_inches="tight", dpi=600 )



plt.figure( "final ion count", figsize=( 12, 9 ) )
plt.xlabel( "Ion Concentration [mol / L]", size=30 )
plt.ylabel( "MS Intensity [a.u.]", size=30 )
plt.xticks( fontsize= 20 )
plt.yticks( fontsize= 20 )
plt.xscale( "log" )
plt.plot( 10**molar_conc, final_ion_count/np.max( final_ion_count )/3.8, color='r', linewidth=2 )
### comparison with experimental data
plt.xlim( [0.8e-6, 2e-1] )
#expdata = np.loadtxt( "experimentell/count_to_conc.txt" )
#plt.plot( 10**expdata[:,0], expdata[:,2]*0.95, color="blue", linestyle=' ', marker='^', ms=15 )
cfile = np.loadtxt(  "experimentell/count_to_conc.txt" )
normCFile = cfile
for i in range( 4 ):
    normCFile[:,i+1] = cfile[:,i+1]/np.sum( cfile[:,i+1] )
avgData = np.zeros( [6,2] )
for i in range( 6 ):
    avgData[i,0] = np.mean(  normCFile[i,1:] )
    avgData[i,1] = np.std(  normCFile[i,1:] )
plt.errorbar( 10**cfile[:,0], avgData[:,0]-0.025, avgData[:,1], elinewidth=2, capsize=5, capthick=2, linestyle='', marker='x', markersize=8, color='b' )
if A2YN == 0:
    plt.savefig( "CtcODE_IonYield.png", bbox_inches="tight" )
    plt.savefig( "CtcODE_IonYield.pdf", bbox_inches="tight", dpi=600 )
    np.savetxt( "CtcODE_IonYield.txt", np.concatenate( ( 10**molar_conc, final_ion_count ), axis=1 ) )
if A2YN == 1:
    plt.savefig( "CtcODE_IonYieldCluster.png", bbox_inches="tight" )
    plt.savefig( "CtcODE_IonYieldCluster.pdf", bbox_inches="tight", dpi=600 )
    np.savetxt( "CtcODE_IonYieldCluster.txt", np.concatenate( ( 10**molar_conc, final_ion_count ), axis=1 ) )


#plt.show(  )

