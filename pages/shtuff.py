
import streamlit as st

st.write("""
### Regular Call and Put Options
The prices at time zero of a regular European call option ($c$) and put option ($p$) are given by:
""")
st.latex(r"""
c = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
""")
st.latex(r"""
p = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)
""")
st.write("Where:")
st.latex(r"""
d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}
""")

st.write("""
### Down-and-In Call Option for $H$ $\leq$ $K$
""")
st.latex(r"""
c_{di} = S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(y) - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(y - \sigma\sqrt{T})
""")
st.latex(r"""
\text{Where: } 
\lambda = \frac{r - q + \sigma^2/2}{\sigma^2}, \quad
y = \frac{\ln(H^2 / (S_0 K))}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}
""")

st.write("""
### Down-and-Out Call Option for $H$ $\leq$ $K$
""")
st.latex(r"""
c_{do} = c - c_{di}
""")

st.write("""
### Down-and-Out Call Option for $H$ $\geq$ $K$
""")

st.latex(r"""
c_{do} = S_0 N(x_1) e^{-qT} - K e^{-rT} N(x_1 - \sigma\sqrt{T}) 
        - S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(y_1) 
        + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(y_1 - \sigma\sqrt{T})
""")
st.latex(r"""
\text{Where: }
x_1 = \frac{\ln(S_0 / H)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}, \quad
y_1 = \frac{\ln(H / S_0)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}
""")

st.write("""
### Down-and-In Call Option for $H$ $\geq$ $K$
""")

st.latex(r"""
c_{di} = c - c_{do}
""")


st.write("""
### Up-and-In Call Option for $H$ $\geq$ $K$

""")
st.latex(r"""
c_{ui} = S_0 N(x_1) e^{-qT} - K e^{-rT} N(x_1 - \sigma\sqrt{T}) 
        - S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda}[N(-y) - N(-y_1)] + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} [N(-y + \sigma\sqrt{T} - N(-y_1 + \sigma\sqrt{T})]
""")

st.write("""
### Up-and-Out Call Option for $H$ $\geq$ $K$
""")
st.latex(r"""
c_{uo} = c - c_{ui}
""")

st.write(""" 
        ### Up-and-In Put Option for $H$ $\geq$ $K$
        """)
st.latex(r"""
p_{ui} = -S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(-y) 
        + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(-y + \sigma\sqrt{T})
""")

st.write(""" 
        ### Up-and-Out Put Option for $H$ $\geq$ $K$
        """)

st.latex(r"""
p_{uo} = p - p_{ui}
""")

st.write(""" 
        ### Up-and-Out Put Option for $H$ $\leq$ $K$
        """)
st.latex(r"""
    p_{uo} = -S_0 N(-x_1) e^{-qT} + K e^{-rT} N(-x_1 + \sigma\sqrt{T}) 
        + S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(-y_1) 
        - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(-y_1 + \sigma\sqrt{T})
        """)

st.write(""" 
        ### Up-and-In Put Option for $H$ $\leq$ $K$
        """)
st.latex(r"""
p_{ui} = p - p_{uo}
""")

st.write(""" 
        ### Down-and-Out Put Option for $H$ $\geq$ $K$
        """)
st.latex(r"""
p_{do} = 0
""")

st.write(""" 
        ### Down-and-In Put Option for $H$ $\geq$ $K$
        """)
st.latex(r"""
p_{di} = p
""")

st.write(""" 
        ### Down-and-In Put Option for $H$ $\leq$ $K$
        """)

st.latex(r"""
    p_{uo} = -S_0 N(-x_1) e^{-qT} + K e^{-rT} N(-x_1 + \sigma\sqrt{T}) 
        + S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} [N(y) - N(y_1)] - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda - 2} [N(y - \sigma\sqrt{T}) - N(y_1 - \sigma\sqrt{T})]
        """)

st.write(""" 
        ### Down-and-Out Put Option for $H$ $\leq$ $K$
        """)
st.latex(r"""
p_{do} = p - p_{di}
""")