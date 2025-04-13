# import streamlit as st

# def display_usage_instructions():
#     # This is your HTML + CSS. 
#     html_code = """
#     <html>
#     <head>
#         <style>
#             body {
#                 font-family: 'Segoe UI', sans-serif;
#                 color: #333;
#             }
#             .guide-container {
#                 max-width: 850px;
#                 margin: 0 auto;
#                 background-color: #ffffff;
#                 border-radius: 16px;
#                 padding: 30px 35px;
#                 box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
#                 border: 1px solid #e0e0e0;
#             }
#             .guide-title {
#                 color: #0066cc;
#                 font-size: 1.8rem;
#                 margin-bottom: 1.2rem;
#                 font-weight: 700;
#             }
#             p, li {
#                 font-size: 1rem;
#                 line-height: 1.6;
#             }
#             .guide-list {
#                 padding-left: 1.2rem;
#             }
#             .guide-list li {
#                 margin-bottom: 0.5rem;
#             }
#             ul {
#                 list-style-type: disc;
#                 padding-left: 1.5rem;
#             }
#             ol {
#                 padding-left: 1.4rem;
#             }
#             .highlight {
#                 background: #e8f0fe;
#                 padding: 0.2rem 0.4rem;
#                 border-radius: 5px;
#                 font-family: monospace;
#             }
#             a {
#                 color: #0066cc;
#                 text-decoration: none;
#                 font-weight: 500;
#             }
#             a:hover {
#                 text-decoration: underline;
#             }
#             .section-icon {
#                 margin-right: 0.4rem;
#             }
#             h3 {
#                 margin-top: 1.5rem;
#                 margin-bottom: 0.5rem;
#                 font-size: 1.2rem;
#                 color: #333;
#             }
#         </style>
#     </head>
#     <body>
#         <div class="guide-container">
#             <h2 class="guide-title">🚀 Quick-Start Guide: Barrier Options Interface</h2>

#             <h3>🔗 1. Accessing the Application</h3>
#             <ol class="guide-list">
#                 <li><em>Open the web page:</em> Navigate to 
#                     <a href="https://barrier-options.streamlit.app/" target="_blank">
#                         https://barrier-options.streamlit.app/
#                     </a>
#                 </li>
#                 <li>The app may take a few moments to load.</li>
#             </ol>

#             <h3>📝 2. Input Parameters</h3>
#             <ol class="guide-list">
#                 <li><strong>Select the Option Type:</strong> Choose knock-in or knock-out, up or down (e.g. <span class="highlight">“Down-and-In Call”</span>).</li>
#                 <li><strong>Enter Contract Details:</strong>
#                     <ul>
#                         <li>Spot Price (<span class="highlight">S0</span>)</li>
#                         <li>Strike Price (<span class="highlight">K</span>)</li>
#                         <li>Barrier Level (<span class="highlight">H</span>)</li>
#                         <li>Time to Maturity (<span class="highlight">T</span>)</li>
#                         <li>Volatility (<span class="highlight">σ</span>)</li>
#                         <li>Risk-free Rate (<span class="highlight">r</span>)</li>
#                         <li>Dividend Yield (<span class="highlight">q</span>, if applicable)</li>
#                     </ul>
#                 </li>
#                 <li><strong>Simulation / Grid Settings:</strong>
#                     <ul>
#                         <li>Number of Paths (<span class="highlight">N</span>)</li>
#                         <li>Number of Time Steps (<span class="highlight">M</span>)</li>
#                         <li>Enable Variance Reduction (e.g. antithetic, control variates)</li>
#                         <li>Toggle Adaptive Mesh Refinement (AMR)</li>
#                     </ul>
#                 </li>
#             </ol>

#             <h3>⚙️ 3. Running the Pricing</h3>
#             <ol class="guide-list">
#                 <li>Click “Run” or “Calculate” to start pricing with selected parameters.</li>
#                 <li>The app will display the <strong>estimated option price</strong> and metrics (e.g. standard error).</li>
#                 <li>Graphical output may include:
#                     <ul>
#                         <li>Monte Carlo price paths</li>
#                         <li>Finite difference solution surfaces</li>
#                         <li>Binomial or AMR lattice structures</li>
#                     </ul>
#                 </li>
#             </ol>

#             <h3>📊 4. Interpreting Results</h3>
#             <ol class="guide-list">
#                 <li><strong>Option Price:</strong> The calculated fair value based on your inputs.</li>
#                 <li><strong>Barrier Events:</strong> Check if the barrier was triggered (knock-in/out).</li>
#                 <li><strong>Sensitivity:</strong> Adjust <span class="highlight">σ</span>, <span class="highlight">H</span>, or <span class="highlight">N</span> to observe effects.</li>
#             </ol>

#             <h3>💡 5. Tips & Best Practices</h3>
#             <ul class="guide-list">
#                 <li>Use small <span class="highlight">N</span> or <span class="highlight">M</span> to test quickly; increase for better accuracy.</li>
#                 <li>Avoid extreme inputs that can lead to numerical instability or long runtimes.</li>
#                 <li>Compare multiple pricing methods to evaluate performance trade-offs.</li>
#             </ul>
#         </div>
#     </body>
#     </html>
#     """


#     st.markdown(html_code, unsafe_allow_html=True)

# def main():
#     st.title("Barrier Options: How to Use This Interface")
#     st.write("Below is a short guide on how to navigate and use the Barrier Options interface.")

#     display_usage_instructions()

# if __name__ == "__main__":
#     main()

import streamlit as st

def display_usage_instructions():
    html_code = """
    <html>
    <head>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                color: #333;
            }
            .guide-container {
                max-width: 850px;
                margin: 0 auto;
                background-color: #ffffff;
                border-radius: 16px;
                padding: 30px 35px;
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
                border: 1px solid #e0e0e0;
            }
            .guide-title {
                color: #0066cc;
                font-size: 1.8rem;
                margin-bottom: 1.2rem;
                font-weight: 700;
            }
            p, li {
                font-size: 1rem;
                line-height: 1.6;
            }
            .guide-list {
                padding-left: 1.2rem;
            }
            .guide-list li {
                margin-bottom: 0.5rem;
            }
            ul {
                list-style-type: disc;
                padding-left: 1.5rem;
            }
            ol {
                padding-left: 1.4rem;
            }
            .highlight {
                background: #e8f0fe;
                padding: 0.2rem 0.4rem;
                border-radius: 5px;
                font-family: monospace;
            }
            a {
                color: #0066cc;
                text-decoration: none;
                font-weight: 500;
            }
            a:hover {
                text-decoration: underline;
            }
            .section-icon {
                margin-right: 0.4rem;
            }
            h3 {
                margin-top: 1.5rem;
                margin-bottom: 0.5rem;
                font-size: 1.2rem;
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="guide-container">
            <h2 class="guide-title">🚀 Quick-Start Guide: Barrier Options Interface</h2>
            <h3>🔗 1. Accessing the Application</h3>
            <ol class="guide-list">
                <li><em>Open the web page:</em> Navigate to 
                    <a href="https://barrier-options.streamlit.app/" target="_blank">
                        https://barrier-options.streamlit.app/
                    </a>
                </li>
                <li>The app may take a few moments to load.</li>
            </ol>
            <h3>📝 2. Input Parameters</h3>
            <ol class="guide-list">
                <li><strong>Select the Option Type:</strong> Choose knock-in or knock-out, up or down (e.g. <span class="highlight">“Down-and-In Call”</span>).</li>
                <li><strong>Enter Contract Details:</strong>
                    <ul>
                        <li>Spot Price (<span class="highlight">S0</span>)</li>
                        <li>Strike Price (<span class="highlight">K</span>)</li>
                        <li>Barrier Level (<span class="highlight">H</span>)</li>
                        <li>Time to Maturity (<span class="highlight">T</span>)</li>
                        <li>Volatility (<span class="highlight">σ</span>)</li>
                        <li>Risk-free Rate (<span class="highlight">r</span>)</li>
                        <li>Dividend Yield (<span class="highlight">q</span>, if applicable)</li>
                    </ul>
                </li>
                <li><strong>Simulation / Grid Settings:</strong>
                    <ul>
                        <li>Number of Paths (<span class="highlight">N</span>)</li>
                        <li>Number of Time Steps (<span class="highlight">M</span>)</li>
                        <li>Enable Variance Reduction (e.g. antithetic, control variates)</li>
                        <li>Toggle Adaptive Mesh Refinement (AMR)</li>
                    </ul>
                </li>
            </ol>
            <h3>⚙️ 3. Running the Pricing</h3>
            <ol class="guide-list">
                <li>Click “Run” or “Calculate” to start pricing with selected parameters.</li>
                <li>The app will display the <strong>estimated option price</strong> and metrics (e.g. standard error).</li>
                <li>Graphical output may include:
                    <ul>
                        <li>Monte Carlo price paths</li>
                        <li>Finite difference solution surfaces</li>
                        <li>Binomial or AMR lattice structures</li>
                    </ul>
                </li>
            </ol>
            <h3>📊 4. Interpreting Results</h3>
            <ol class="guide-list">
                <li><strong>Option Price:</strong> The calculated fair value based on your inputs.</li>
                <li><strong>Barrier Events:</strong> Check if the barrier was triggered (knock-in/out).</li>
                <li><strong>Sensitivity:</strong> Adjust <span class="highlight">σ</span>, <span class="highlight">H</span>, or <span class="highlight">N</span> to observe effects.</li>
            </ol>
            <h3>💡 5. Tips & Best Practices</h3>
            <ul class="guide-list">
                <li>Use small <span class="highlight">N</span> or <span class="highlight">M</span> to test quickly; increase for better accuracy.</li>
                <li>Avoid extreme inputs that can lead to numerical instability or long runtimes.</li>
                <li>Compare multiple pricing methods to evaluate performance trade-offs.</li>
            </ul>
        </div>
    </body>
    </html>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def main():
    st.title("Barrier Options: How to Use This Interface")
    st.write("Below is a short guide on how to navigate and use the Barrier Options interface.")
    display_usage_instructions()

if __name__ == "__main__":
    main()
