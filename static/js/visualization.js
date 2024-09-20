function renderVisualization(data){
    if (data === undefined || Object.keys(data).length === 0){
        return;
    }
    // after obtaining the return
    // n: degree of polynomial

    const cur_input_poly = document.getElementById('input_poly').value.trim();
    if (cur_input_poly !== data.txt.trim()){
        document.getElementById("input_poly").value = data.txt;
    }
    // document.getElementById("rootsinfo").innerHTML = '';

    // set global variables
    sos_poly = data.txt;

    // First render the coefficients
    renderCoeffs(data.n, data.triangle, data.gens);

    // Next render the heatmap triangle
    renderHeatmap(data.heatmap);
}

function _get_first_three_gens(all_gens = 'abc'){
    let gens = ['a','b','c'];
    for (let i = 0; i < Math.min(all_gens.length, 3); ++i){
        gens[i] = all_gens[i];
    }
    return gens;
}



function renderCoeffs(degree, triangle_vals, all_gens = 'abc'){
    // First render the coefficient triangle
    let coeffs_view = document.getElementById('coeffs');
    // if (degree != degree){
    for (let i=sos_coeffs.length-1;i>=0;--i){
        coeffs_view.removeChild(sos_coeffs.pop());
    }
    // }

    let t = 0;
    const l = 100. / (1 + degree); // length of each samll equilateral triangle 
    const ulx = (50 - l * degree / 2), uly = (29 - 13 * degree * l / 45);
    const fontsize = Math.max(11, Math.floor(28 - 1.5*degree));
    const lengthscale = .28 * fontsize / 20.;
    const titleparser = (chr, deg) => { 
        return deg <= 0? '': (deg==1? chr: chr+deg);
    }

    const gens = _get_first_three_gens(all_gens);
    for (let i = 0; i <= degree; ++i){
        for (let j = 0; j <= i; ++j){
            let coeff = document.createElement('p');
            coeffs_view.appendChild(coeff);
            coeff.innerHTML = triangle_vals[t];
            let x =  (ulx + l*(2*i-j)/2 - triangle_vals[t].length * lengthscale - 2*Math.floor(triangle_vals[t].length/4)), 
                y =  (uly + l*j*13/15);
            coeff.setAttribute('style','position:absolute;left:'+x+'%;top:'+y+'%;font-size:'
                            +fontsize+'px');
            // when mouse hover on a monom, it shows hint like 'a5c' for a^5 * c
            coeff.title = titleparser(gens[0],degree-i) + titleparser(gens[1],i-j) + titleparser(gens[2],j);
            coeff.onmouseover = () => {coeff.style.color = 'blue';}
            if (triangle_vals[t] === '0'){
                coeff.style.color = 'rgb(180,180,180)';
                coeff.onmouseout  = () => {coeff.style.color = 'rgb(180,180,180)';}
            }else{
                coeff.style.color = 'black';
                coeff.onmouseout  = () => {coeff.style.color = 'black';}
            }
            sos_coeffs.push(coeff)
            ++t;
        }
    }
}

function renderHeatmap(heatmap){
    let t = 0;
    for (let i=0;i<=heatmap_gridsize;++i){
        for (let j=0;j<=heatmap_gridsize-i;++j){
            heatmap_grids[t].style.background = 'rgb('
                +heatmap[t][0]+','+heatmap[t][1]+','+heatmap[t][2]+')';
            t += 1;
        }
    }
}

function setSOSResult(data){
    /*
    Set the result of the SOS calculation to the page.
    */
    if (data === undefined){
        return;
    }

    // write the result to the current page
    sos_results.success = data.success;
    sos_results.latex = data.latex; 
    sos_results.txt   = data.txt;
    sos_results.formatted = data.formatted;
    document.getElementById('output_result').innerHTML = data[sos_results.show];

    if (data.success){
        const index = _history_timestamp_to_ind[data.timestamp];
        if (index !== undefined){
            history_data[index].div.style.color = COLOR_SOS_SUCCESS;
        }
    }

    // remove the previous result
    const shadow_box = document.getElementById("shadow_box");
    let past;
    if ((past = shadow_box.firstChild)){
        past.remove();
    }
    
    let str;
    if (!data.success){
        str = '\\quad\\quad\\quad{\\rm Failed}\\quad\\quad\\quad';
    }else{
        const text_length = sos_results.latex.length;
        str = sos_results.latex.slice(2, text_length-2);
        if (str.indexOf('aligned') < 0){ // no aligned environment
            if (str.indexOf('\\\\') >= 0){
                // use \begin{aligned} when there is a line break
                let i = 0;
                while (i+1 < str.length){
                    if (str[i] == '\\' && str[i+1] == '\\'){
                        str = str.slice(0, i+2) + ' & ' + str.slice(i+2, str.length);
                        i += 2;        
                    }
                    ++i;
                }
                str = '\\begin{aligned}  \\  &' + str + '\\end{aligned}';
            }
        }
    }

    // render the new result
    let svg = window.MathJax.tex2svg(str).children[0];
    // shadow_box.style.height = '80%';
    shadow_box.appendChild(svg);
    svg.setAttribute('style', 'width: 95%; height: 90%; margin-top: 2%'); 
    shadow_box.hidden = "";
    return shadow_box;
}