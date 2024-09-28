/*
This file contains the functions for rendering the visualization of the polynomial.
*/

let _3d_vis = {'in3d': false};
// const VIEW3DPARENT = document.getElementById('centerscreen');
const VIEW3DPARENT = document.getElementById('coeffs');
const VIEW2DPARENT = document.getElementById('coeffs');
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
    // console.log(data.triangle.length, n, data.triangle);
    
    const coeffs_view = _3d_vis.in3d?VIEW3DPARENT:VIEW2DPARENT;
    for (let i=sos_coeffs.length-1;i>=0;--i){
        coeffs_view.removeChild(sos_coeffs.pop());
    }

    if (data.gens.length === 3){
        renderCoeffs(data.n, data.triangle, data.gens);
    }else if (data.gens.length === 4){
        renderCoeffs3D(data.n, data.triangle, data.gens);
    }

    if (_3d_vis.in3d){
        // VIEW2DPARENT.hidden = true;
        _3d_vis.tetrahedron.visible = true;
    }else{
        // VIEW2DPARENT.hidden = false;
        _3d_vis.tetrahedron.visible = false;
    }
    // _3d_vis.controls.enableDamping = _3d_vis.in3d;
    _3d_vis.controls.enableZoom = _3d_vis.in3d;
    _3d_vis.controls.enablePan = _3d_vis.in3d;
    _3d_vis.controls.enableRotate = _3d_vis.in3d;
    // _3d_vis.controls.update();

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
    _3d_vis.in3d = false;
    let coeffs_view = VIEW2DPARENT;
    coeffs_view.hidden = false;
    // if (degree != degree){
    // }

    let t = 0;
    const l = 100. / (1 + degree); // length of each samll equilateral triangle 
    const ulx = (50 - l * degree / 2);
    const uly = (29 - 13 * degree * l / 45) + coeffs_view.clientHeight * 0.0015; // margin top
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


/*
Below contains 3d visualization for 4var polynomials
*/
const LABELMARGIN = 20;
const COLLIDE_SCALE = 1.01;
function init3DVisualization(){// Get the div element
    const coeffs_view = VIEW3DPARENT;
    
    // Create a scene
    const scene = new THREE.Scene();
    // scene.background = new THREE.Color(0xffffff);

    // Create a camera
    const camera = new THREE.PerspectiveCamera(75, coeffs_view.clientWidth / coeffs_view.clientHeight, 0.1, 1000);
    const cameraViewProjectionMatrix = new THREE.Matrix4();
    // const frustum = new THREE.Frustum();
    camera.position.z = 5;
    
    // Create a WebGL renderer and attach it to the div
    const renderer = new THREE.WebGLRenderer({alpha:true});
    renderer.setSize(coeffs_view.clientWidth, coeffs_view.clientHeight);
    coeffs_view.appendChild(renderer.domElement);


    // Add orbit controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    const raycaster = new THREE.Raycaster();

    // Add a tetrahedron or other objects to the scene
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([
        1, 1, 1,   // Vertex 0
        -1, -1, 1, // Vertex 1
        -1, 1, -1, // Vertex 2
        1, -1, -1  // Vertex 3
    ]);
    const indices = new Uint16Array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]);
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    
    const material = new THREE.MeshBasicMaterial({
        color: 0x0077ff, 
        wireframe: true, 
        opacity: 1, 
        side: THREE.DoubleSide
    });
    const tetrahedron = new THREE.Mesh(geometry, material);
    tetrahedron.visible = false;
    scene.add(tetrahedron);
    
    _3d_vis.scene = scene;
    _3d_vis.camera = camera;
    _3d_vis.renderer = renderer;
    _3d_vis.controls = controls;
    _3d_vis.raycaster = raycaster;
    _3d_vis.tetrahedron = tetrahedron;
    _3d_vis.positions = [];
    // _3d_vis.frustum = frustum;
    _3d_vis.cameraViewProjectionMatrix = cameraViewProjectionMatrix;
    _3d_vis.tetrav = [
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ];

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        if (_3d_vis.in3d){
            updateTextPositions();
        }
        renderer.render(scene, camera);
    }
    animate();
    
}
init3DVisualization();


function updateTextPositions() {
    if (sos_coeffs.length === 0){
        return;
    }
    const coeffs_view = VIEW3DPARENT;
    const tetrav = _3d_vis.tetrav;
    const camera = _3d_vis.camera;
    camera.updateMatrixWorld(); // Ensure the camera's world matrix is updated
    _3d_vis.cameraViewProjectionMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    // _3d_vis.frustum.setFromProjectionMatrix(_3d_vis.cameraViewProjectionMatrix);
    const cameraPos = camera.position;
    const positions = _3d_vis.positions;
    const raycaster = _3d_vis.raycaster;
    const tetrahedron = _3d_vis.tetrahedron;

    const rect = coeffs_view.getBoundingClientRect();
    const widthHalf = rect.width/2; //coeffs_view.clientWidth / 2;
    const heightHalf = rect.height/2; //coeffs_view.clientHeight / 2;
    sos_coeffs.forEach((textElement, index) => {
        const pos = positions[index];
        // console.log(pos);

        // Create a ray from the camera to the point

        // Project 3D point to 2D screen
        const vector = new THREE.Vector3(pos.x,pos.y,pos.z).project(camera);
        const x = (vector.x * widthHalf + widthHalf);
        const y = (-(vector.y * heightHalf) + heightHalf);
        textElement.style.left = x + 'px';
        textElement.style.top = y + 'px';
        if (x < LABELMARGIN || x > coeffs_view.clientWidth-LABELMARGIN || y < LABELMARGIN || y > coeffs_view.clientHeight-LABELMARGIN){
            textElement.style.display = 'none';
        }
        else{
            textElement.style.display = 'block';
        
            const ray = new THREE.Vector3(pos.x*COLLIDE_SCALE,pos.y*COLLIDE_SCALE,pos.z*COLLIDE_SCALE).sub(cameraPos).normalize();
            raycaster.set(camera.position, ray);
            const intersections = raycaster.intersectObject(tetrahedron);
            const isInside = intersections.length > 0 && intersections[0].distance < cameraPos.distanceTo(
                new THREE.Vector3(pos.x*COLLIDE_SCALE,pos.y*COLLIDE_SCALE,pos.z*COLLIDE_SCALE));
            textElement.style.opacity = isInside ? 0.2 : 1; // Fully opaque if inside, semi-transparent if not
        }
    });
}

function _generateMonoms(nvars, degree) {
    /* Generate all monomials of degree `degree` in `nvars` variables. Sorted in decreasing order. */
    function generateTuples(currentTuple, currentSum, remainingVars) {
        if (remainingVars === 1) {
            return [[...currentTuple, degree - currentSum]];
        } else {
            const tuples = [];
            for (let i = degree - currentSum; i >= 0; i--) {
                tuples.push(...generateTuples([...currentTuple, i], currentSum + i, remainingVars - 1));
            }
            return tuples;
        }
    }

    return generateTuples([], 0, nvars);
}

function renderCoeffs3D(degree, triangle_vals, all_gens = 'abcd'){
    // First render the coefficient triangle
    if (degree === 0){
        return;
    }
    const coeffs_view = VIEW3DPARENT;
    let positions = [];
    const tetrav = _3d_vis.tetrav;
    const inv_degree = 1./degree;
    const titleparser = (chr, deg) => { 
        return deg <= 0? '': (deg==1? chr: chr+deg);
    }
    _generateMonoms(4, degree).forEach((monom, i) => {
        const pos = new THREE.Vector3(
            (monom[0]*tetrav[0][0] + monom[1]*tetrav[1][0] + monom[2]*tetrav[2][0] + monom[3]*tetrav[3][0])*inv_degree,
            (monom[0]*tetrav[0][1] + monom[1]*tetrav[1][1] + monom[2]*tetrav[2][1] + monom[3]*tetrav[3][1])*inv_degree,
            (monom[0]*tetrav[0][2] + monom[1]*tetrav[1][2] + monom[2]*tetrav[2][2] + monom[3]*tetrav[3][2])*inv_degree
        );
        positions.push(pos);
        _3d_vis.positions = positions;
        
        const coeff = document.createElement('p');
        coeff.innerText = triangle_vals[i];
        coeff.className = 'coeff-label3d'; // Add a class name
        coeff.style.position = 'absolute';
        coeff.style.color = 'black';
        coeff.style.pointerEvents = 'auto';
        coeff.style.userSelect = 'none';
        
        // when mouse hover on a monom, it shows hint like 'a5c' for a^5 * c
        coeff.title = titleparser(all_gens[0],monom[0])+titleparser(all_gens[1],monom[1])+titleparser(all_gens[2],monom[2])+titleparser(all_gens[3],monom[3]);
        coeff.onmouseover = () => {coeff.style.color = 'blue';}
        if (false){//(triangle_vals[i] === '0'){
            coeff.style.color = 'rgb(180,180,180)';
            coeff.onmouseout  = () => {coeff.style.color = 'rgb(180,180,180)';}
        }else{
            coeff.style.color = 'black';
            coeff.onmouseout  = () => {coeff.style.color = 'black';}
        }
        coeffs_view.appendChild(coeff);
        sos_coeffs.push(coeff);
    });
    _3d_vis.in3d = true;
}