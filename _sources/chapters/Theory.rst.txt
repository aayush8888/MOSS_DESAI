
Theory
==========================
.. note:: Below - lots of calculations -> Go through at readers risk

PERIODS AFTER INTERACTION
--------------------------

Non-conservative mass transfer:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note:: See Onno Pols lecture notes on binaries for reaching the equation below.

.. math::

    \frac{\dot{a}}{a} = -2 \frac{\dot{M}_d}{M_d} \left( 1 - \beta \frac{M_d}{M_a} - (1-\beta) (\gamma +\frac{1}{2})\frac{M_d}{M_d + M_a} \right)

It can also be written as

.. math::

    \frac{\dot{a}}{a} = -2\left( \frac{\dot{M}_d}{M_d} + \frac{\dot{M}_a}{M_a} - (\gamma +\frac{1}{2})\frac{\dot{M}}{M} \right)

where :math:`M = M_d + M_a, M_a` is the mass of the accretor, and :math:`M_d` is the mass of the donor. This is because :math:`\dot{M}_d = - \beta \dot{M}_a` and therefore also :math:`\dot{M} = \dot{M}_d + \dot{M}_a  = (1-\beta)\dot{M}_d`.

To begin with, we can remove the time dependence:

.. math::

    \frac{\dot{a}}{a} = \frac{da}{a}\frac{1}{dt}

    -2\left( \frac{\dot{M}_d}{M_d} + \frac{\dot{M}_a}{M_a} - (\gamma +\frac{1}{2})\frac{\dot{M}}{M} \right) = -2\left( \frac{dM_d}{M_d} + \frac{dM_a}{M_a} -  (\gamma + \frac{1}{2})\frac{dM}{M} \right)\frac{1}{dt}

    \frac{da}{a} = -2\left( \frac{dM_d}{M_d} + \frac{dM_a}{M_a} -  (\gamma + \frac{1}{2})\frac{dM}{M} \right)

To solve this, we start with the left-hand-side (LHS) of the equation, integrating it:

.. math::

    \int_{a_i}^{a_f}\frac{da}{a} = \int_{\ln a_i}^{\ln  a_f} d\ln a = \ln a_f - \ln a_i = \ln  \left( \frac{a_f}{a_i} \right)

The right hand side (RHS) is also integrated

.. math::

    -2\left( \int_{M_{d,i}}^{M_{d,f}} \frac{dM_d}{M_d} + \int_{M_{a,i}}^{M_{a,f}} \frac{dM_a}{M_a} - \int_{M_i}^{M_f} (\gamma + \frac{1}{2})\frac{dM}{M} \right)

Assuming that :math:`\gamma` is a constant gives

.. math::

    -2\left( \int_{\ln M_{d,i}}^{\ln M_{d,f}} d\ln M_d + \int_{\ln M_{a,i}}^{\ln M_{a,f}} d\ln M_a - (\gamma + \frac{1}{2}) \int_{\ln M_i}^{\ln M_f} d\ln M \right)

    -2\left(\ln M_{d,f} - \ln M_{d,i} + \ln M_{a,f} - \ln M_{a,i} - (\gamma + \frac{1}{2})(\ln M_f -  \ln M_i)\right)

    -2 \left( \ln \frac{M_{d,f} M_{a,f}}{M_{d,i} M_{a,i}} - (\gamma + \frac{1}{2})\ln \frac{M_f}{M_i} \right)

    \ln \left(\frac{M_{d,f} M_{a,f}}{M_{d,i} M_{a,i}}\right)^{-2} + \ln \left(\frac{M_{d,f} + M_{a,f}}{M_{d,i} + M_{a,i}}\right)^{2\gamma+1}

    \ln \left( \left(\frac{M_{d,i} M_{a,i}}{M_{d,f} M_{a,f}}\right)^{2} \left(\frac{M_{d,f} + M_{a,f}}{M_{d,i} + M_{a,i}}\right)^{2\gamma+1}  \right)

Putting the LHS and RHS together means that:

.. math::

    \frac{a_f}{a_i} = \left(\frac{M_{d,i} M_{a,i}}{M_{d,f} M_{a,f}}\right)^{2} \left(\frac{M_{d,f} + M_{a,f}}{M_{d,i} + M_{a,i}}\right)^{2\gamma+1}

Isotropic re-emission
~~~~~~~~~~~~~~~~~~~~~~
In case the mass transfer efficiency is 1 (:math:`\beta = 1`), the :math:`dM` term disappears, meaning that it is independent on whether :math:`\gamma` is a constant or not. Then

.. math::

    \frac{a_f}{a_i} =  \left( \frac{M_{d,i}M_{a,i}}{M_{d,f}M_{a,f}} \right)^2

I case the mass transfer was completely non-conservative (:math:`\beta = 0`), then M_a is a constant. That means that we can account for isotropic re-emission and set :math:`\gamma = M_d/M_a = (M - M_a)/M_a = M/M_a - 1`. This results in:

.. math::

    \int _{M_i}^{M_f} (\gamma + \frac{1}{2}) \frac{dM}{M} = \int _{M_i}^{M_f} (\frac{M}{M_a} - \frac{1}{2}) \frac{dM}{M} 
    
    = \frac{1}{M_a} \int_{M_i}^{M_f} dM - \frac{1}{2}\int_{\ln M_i}^{\ln M_f} d\ln M = \frac{M_f - M_i}{M_a} + \ln (M_i/M_f)^{1/2}

which means that then

.. math::

    \ln \frac{a_f}{a_i} = \ln \left( \frac{M_{d,i}M_{a,i}}{M_{d,f}M_{a,f}} \right)^2 + 2\frac{M_f - M_i}{M_a} + \ln (M_i/M_f)


Circumbinary ring
~~~~~~~~~~~~~~~~~~
In the case of a circumbinary ring, the :math:`\gamma = \frac{(M_d + M_a)^2}{M_d M_a} \sqrt{\frac{a_{\rm ring}}{a}}.` Following Artymowicz :math:`\&` Lubow 1994, we set :math:`a_{ring} = 2a` and get :math:`\gamma = \sqrt{2}\frac{(M_d + M_a)^2}{M_d M_a}` (see also Zapartas et al. 2017a). 

For fully non-conservative mass transfer (:math:`\beta = 0`), :math:`M_a` is constant. This means that the RHS becomes

.. math::

    -2\left( \int_{M_{d,i}}^{M_{d,f}} \frac{dM_d}{M_d} + \int_{M_{a,i}}^{M_{a,f}} \frac{dM_a}{M_a} - \int_{M_i}^{M_f} (\gamma + \frac{1}{2})\frac{dM}{M} \right) 
    
    = -2 \ln \left( \frac{M_{d,f}}{M_{d,i}} \right) + 2 \int _{M_i}^{M_f} \gamma \frac{dM}{M} + \int _{M_i}^{M_f} \frac{dM}{M} 
    
    = -2 \ln \left( \frac{M_{d,f}}{M_{d,i}} \right) + \frac{2\sqrt{2}}{M_a} \int _{M_i}^{M_f} \frac{M}{(M-M_a)} dM + \ln \left( \frac{M_f}{M_i} \right) 
    
    = -2 \ln \left( \frac{M_{d,f}}{M_{d,i}} \right) + \frac{2\sqrt{2}}{M_a} \left( M_a \ln \left( \frac{M_f - M_a}{M_i-M_a} \right) + M_f - M_i \right) + \ln \left( \frac{M_f}{M_i} \right) 

And can then be equaled with the left-hand side

For fully conservative mass transfer, the integral with the :math:`\gamma` disappears since M is a constant. The separation is then calculated in the same way as for the other cases:

.. math::

    \frac{a_f}{a_i} =  \left( \frac{M_{d,i}M_{a,i}}{M_{d,f}M_{a,f}} \right)^2


Fast wind
Should I calculate this? Is this necessary? (:math:`\gamma = M_a/M_d`)

Step from separation to period

And, finally, the separation :math:`a` can be translated to a period using Kepler III:

.. math::

    \frac{P^2}{a^3} = \frac{4\pi}{G(M_d + M_a)}

where the gravitational constant :math:`G = 4\pi  AU^3  yr^{-2} M_{\odot}^{-1}`.

REJUVENATION
------------
 
From Tout et al. (1997) there is in Section 5.1 a treatment for rejuvenation.

.. math::

    t' = \dfrac{\mu}{\mu '}\dfrac{\tau_{\text{MS}} '}{\tau_{\text{MS}}} t

where :math:`t'` is the apparent age of the star right after mass accretion, t is the apparent age of the star if it wouldn't have been rejuvenated, :math:`\tau_{\text{MS}}` is the main sequence lifetime of the accretor prior to accretion, :math:`\tau_{\text{MS}}'` is the main sequence lifetime for a star with the initial mass that is the same of the accretor after mass accretion. The parameters :math:`\mu` are included when the accretor has a convective core and are then :math:`\mu = M_2` and :math:`\mu ' = M_2  '`.

This means that a star that is rejuvenated lives t-t' years in addition to the new assumed lifetime of the star.