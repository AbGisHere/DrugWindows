import base64
import json

def show_structure(protein_text: str = None, ligand_text: str = None, pdb_id: str = "Structure", protein_name: str = "") -> str:
    """
    Robust 3D visualization with:
    1. Dark/Black Background
    2. Constant Protein Color (Bright Light Blue)
    3. High-Contrast Ligands (Magenta)
    """
    
    # 1. Safely serialize inputs
    prot_json = json.dumps(protein_text)
    lig_json = json.dumps(ligand_text)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            /* 1. Set Main Background to Black */
            body {{ margin: 0; padding: 0; overflow: hidden; background-color: #000000; }}
            #container {{ width: 100vw; height: 100vh; position: relative; }}
            
            #error-log {{ 
                display: none; position: absolute; top: 10px; left: 10px; 
                background: rgba(255,0,0,0.8); color: white; padding: 10px; 
                z-index: 999; font-family: sans-serif; border-radius: 5px;
            }}
            
            /* 2. Update Legend for Dark Mode */
            .legend {{
                position: absolute; bottom: 10px; right: 10px;
                background: rgba(30, 30, 30, 0.9);
                color: #ffffff;
                padding: 8px 12px;
                border: 1px solid #444;
                border-radius: 6px; font-family: sans-serif; font-size: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.5); z-index: 100;
                pointer-events: none;
            }}
            .color-box {{ 
                display: inline-block; width: 10px; height: 10px; 
                margin-right: 6px; border-radius: 50%; 
            }}
            
            /* 3. Update Info Overlay for Dark Mode */
            .info-overlay {{
                position: absolute; top: 10px; left: 10px;
                background: rgba(30, 30, 30, 0.9);
                color: #ffffff;
                padding: 8px 12px;
                border: 1px solid #444;
                border-radius: 6px; font-family: sans-serif; font-size: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.5); z-index: 90;
            }}
        </style>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div id="error-log"></div>
        
        <div class="info-overlay">
            <strong>{pdb_id}</strong><br>
            <span style="font-size:12px; color:#aaa">{protein_name}</span>
        </div>

        <div class="legend">
            <div><span class="color-box" style="background: #33CCFF;"></span>Protein</div>
            <div style="margin-top:4px"><span class="color-box" style="background: magenta;"></span>Ligand</div>
        </div>

        <div id="container"></div>

        <script>
            function logError(msg) {{
                var el = document.getElementById('error-log');
                el.style.display = 'block';
                el.innerHTML += "Error: " + msg + "<br>";
                console.error(msg);
            }}

            window.onload = function() {{
                try {{
                    if (typeof $3Dmol === 'undefined') {{
                        throw new Error("3Dmol.js failed to load. Check internet connection.");
                    }}

                    var element = document.getElementById('container');
                    
                    // 4. Set 3Dmol Viewer Background to Black
                    var config = {{ backgroundColor: 'black' }};
                    var viewer = $3Dmol.createViewer(element, config);

                    var proteinData = {prot_json};
                    var ligandData = {lig_json};
                    var hasModel = false;

                    // --- ADD PROTEIN ---
                    if (proteinData) {{
                        viewer.addModel(proteinData, "pdb");
                        
                        // CHANGED: Use a brighter Light Blue (#33CCFF)
                        viewer.setStyle(
                            {{model: -1}}, 
                            {{cartoon: {{color: '#33CCFF'}}}} 
                        );
                        
                        // Hetatoms (non-protein atoms) styling
                        viewer.addStyle(
                            {{model: -1, hetflag: true}}, 
                            {{stick: {{radius: 0.1, color: 'lightgray'}}}}
                        );
                        hasModel = true;
                    }}

                    // --- ADD LIGAND ---
                    if (ligandData) {{
                        viewer.addModel(ligandData, "pdb");
                        viewer.setStyle(
                            {{model: -1}}, 
                            {{stick: {{colorscheme: 'magentaCarbon', radius: 0.4}}}}
                        );
                        hasModel = true;
                    }} else if (proteinData) {{
                        // Fallback: Internal ligands
                        viewer.addStyle(
                            {{resn: ["UNL", "LIG", "DRG", "UNK"]}}, 
                            {{stick: {{colorscheme: 'magentaCarbon', radius: 0.4}}}}
                        );
                    }}

                    if (!hasModel) {{
                        throw new Error("No protein or ligand data provided.");
                    }}

                    viewer.zoomTo();
                    viewer.render();

                }} catch(e) {{
                    logError(e.message);
                }}
            }};
        </script>
    </body>
    </html>
    """
    
    b64 = base64.b64encode(html_content.encode()).decode()
    iframe = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="500" frameborder="0" style="border: 1px solid #333; border-radius: 8px;"></iframe>'
    
    return iframe