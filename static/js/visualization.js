/*
This file contains the functions for rendering the visualization of the polynomial.
*/

let _3d_vis = {'in3d': false, 'heatmap_size_3d': 18};

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
        if (_3d_vis.tetrav.length){
            _3d_vis.tetrahedron.visible = true;
            _3d_vis.pointCloud.visible = true;
        }else{
            _3d_vis.tetrahedron.style.display = 'block';
        }
        renderHeatmap3D(data.heatmap);
    }else{
        // VIEW2DPARENT.hidden = false;
        if (_3d_vis.tetrav.length){
            _3d_vis.tetrahedron.visible = false;
            _3d_vis.pointCloud.visible = false;
        }else{
            _3d_vis.tetrahedron.style.display = 'none';
        }
        renderHeatmap(data.heatmap);
    }
    // _3d_vis.controls.enableDamping = _3d_vis.in3d;
    _3d_vis.controls.enableZoom = _3d_vis.in3d;
    _3d_vis.controls.enablePan = _3d_vis.in3d;
    _3d_vis.controls.enableRotate = _3d_vis.in3d;
    // _3d_vis.controls.update();

    // Next render the heatmap triangle
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
    const hideZeros = !document.getElementById('config_ShowZeros').checked;
    for (let i = 0; i <= degree; ++i){
        for (let j = 0; j <= i; ++j){
            let coeff = document.createElement('p');
            coeff.classList.add('coeff-label');
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
                if (hideZeros){
                    coeff.style.display = 'none';
                }
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
    if ((heatmap === undefined)||(heatmap === null)){
        const WHITE = 'rgb(255,255,255)';
        for (let i=0;i<=heatmap_gridsize;++i){
            for (let j=0;j<=heatmap_gridsize-i;++j){
                heatmap_grids[t].style.background = WHITE;
                t += 1;
            }
        }
        return;
    }
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
    const getValue = (x) => {return x.endsWith('ex')? x.slice(0,-2):100;}
    const svg_height = getValue(svg.getAttribute('height'));
    const svg_width  = getValue(svg.getAttribute('width'));
    shadow_box.appendChild(svg);
    if (svg_width<50&&svg_height<10&&svg_height*svg_width<80){
        svg.setAttribute('style', 'width: 63%; height: 59%; margin-top: 8%;');
    }else {
        svg.setAttribute('style', 'width: 95%; height: 90%; margin-top: 2%;');
    }
    shadow_box.hidden = "";
    return shadow_box;
}


/*
Below contains 3d visualization for 4var polynomials
*/
const LABELMARGIN = 20;
const COLLIDE_SCALE = 1.01;

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
    
    // const material = new THREE.MeshBasicMaterial({
    //     color: 0x0077ff, 
    //     wireframe: true, 
    //     opacity: 1, 
    //     side: THREE.DoubleSide
    // });
    const material = new THREE.MeshBasicMaterial({
        color: 0x000000,     //
        transparent: true,   // Allow translucency
        wireframe: true, 
        opacity: 0.2,        // Set opacity for a light, translucent effect
        // shininess: 10,       // Add slight shininess to give it a glowing effect
        side: THREE.DoubleSide // Make it visible from both sides
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
    _3d_vis.dashlines = null;
    _3d_vis.cameraViewProjectionMatrix = cameraViewProjectionMatrix;
    _3d_vis.tetrav = [
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ];

    // _initDashlinesPlugin(); // enable/disable dashed lines
    _init3DHeatmap();

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        if (_3d_vis.in3d){
            updateTextPositions();

            if (_3d_vis.dashlines !== null){
                const tetrav = _3d_vis.tetrav;
                _3d_vis.dashlines.forEach((line, i) => {
                    const v = new THREE.Vector3(tetrav[i][0], tetrav[i][1], tetrav[i][2]);
                    //i%1==0?new THREE.Vector3(1,1,1):new THREE.Vector3(-1,-1,1);
                    const vision_vector = v.sub(_3d_vis.camera.position).normalize();
                    const normal = _3d_vis.tetrav[3-i]; // this is not Vector3 class, so do not use .dot method
                    line.visible = true;
                    if (vision_vector.x * normal[0] + vision_vector.y * normal[1] + vision_vector.z * normal[2] > 0){
                        line.material = _3d_vis.dashlinesSolidMaterial;
                    }else{
                        line.material = _3d_vis.dashlinesDashedMaterial;
                        line.computeLineDistances(); // dashed lines need distances to be computed
                    }
                });
            }
        }        
        renderer.render(scene, camera);
    }
    animate();

}

function _initDashlinesPlugin(){
    // contributed by 故宫的落日
    // TODO: use 6 solid/dashed lines rather than 4 solid/dashed surfaces
    
    const tetrav = _3d_vis.tetrav;
    const scene = _3d_vis.scene;
    
    const xs_dashed_material = new THREE.LineDashedMaterial({
        color: 0x000000,
        linewidth: 1,
        scale: 1,
        dashSize: 0.1,
        gapSize: 0.1,
        transparent: true,
        opacity: 0.2,
    });

    const xs_solid_material = new THREE.LineBasicMaterial({
        color: 0x333333,
        linewidth: 1,
        transparent: true,
        opacity: 0.2,
    });

    scene.remove(_3d_vis.tetrahedron);
    // _3d_vis.tetrahedron = {visible: false}; // abstract tetrahedron

    const xs_line_fs = [[0,2,1,0],[1,3,0,1],[0,3,2,0],[1,2,3,1]].map(face => {
        return new THREE.Line(new THREE.BufferGeometry().setFromPoints(
            face.map(j => new THREE.Vector3(tetrav[j][0], tetrav[j][1], tetrav[j][2]))
        ), xs_solid_material);
    });
    xs_line_fs.forEach((line, i) => {
        scene.add(line);
        line.visible = false;
    });
    _3d_vis.dashlines = xs_line_fs;
    _3d_vis.dashlinesDashedMaterial = xs_dashed_material;
    _3d_vis.dashlinesSolidMaterial = xs_solid_material;
}


function _init3DHeatmap(){
    
    // Particle geometry for Gaussian splatting
    const particles = new THREE.BufferGeometry();
    const degree = _3d_vis.heatmap_size_3d;
    const inv_degree = 1./degree;
    const tetrav = _3d_vis.tetrav;
    const particle_positions = [];
    const particle_colors = [];
    _generateMonoms(4, degree).forEach((monom, i) => {
        const pos = new THREE.Vector3(
            (monom[0]*tetrav[0][0] + monom[1]*tetrav[1][0] + monom[2]*tetrav[2][0] + monom[3]*tetrav[3][0])*inv_degree,
            (monom[0]*tetrav[0][1] + monom[1]*tetrav[1][1] + monom[2]*tetrav[2][1] + monom[3]*tetrav[3][1])*inv_degree,
            (monom[0]*tetrav[0][2] + monom[1]*tetrav[1][2] + monom[2]*tetrav[2][2] + monom[3]*tetrav[3][2])*inv_degree
        )
        particle_positions.push(pos.x, pos.y, pos.z);
        particle_colors.push(255,255,255);
    });

    particles.setAttribute('position', new THREE.Float32BufferAttribute(particle_positions, 3));
    particles.setAttribute('color', new THREE.Float32BufferAttribute(particle_colors, 3));


    // const particleCount = 1000;
    // const positions = [];
    // const colors = [];
    // for (let i = 0; i < particleCount; i++) {
    //     // Randomize positions within tetrahedron bounds
    //     const pos = new THREE.Vector3(Math.random(),Math.random(),Math.random()); // Function to ensure particles inside the volume
    //     positions.push(pos.x, pos.y, pos.z);

    //     // Assign color based on "heat" value or density at position
    //     const heatValue = Math.random() / 3; // Function that returns heatmap value for color
    //     const color = new THREE.Color().setHSL(0.7 - heatValue * 0.7, 1.0, 0.5); // Adjust HSL for heat color
    //     colors.push(color.r, color.g, color.b);
    // }
    const defaultParticleSize = document.getElementById('config_3D_PointSize').value / 10.0;
    const defaultHeatmapOpacity = document.getElementById('config_3D_HeatmapOpacity').value / 100.0;
    const particleMaterial = new THREE.PointsMaterial({
        size: defaultParticleSize,
        vertexColors: true,
        opacity: defaultHeatmapOpacity,
        transparent: true,
        sizeAttenuation: false
    });

    const pointCloud = new THREE.Points(particles, particleMaterial);
    pointCloud.visible = false;
    _3d_vis.scene.add(pointCloud);
    _3d_vis.pointCloud = pointCloud;


    // event listeners for configuration
    const config_3D_PointSize = document.getElementById('config_3D_PointSize');
    config_3D_PointSize.oninput = function() {
        pointCloud.material.size = (this.value/10.0);
        pointCloud.material.needsUpdate = true;
    }
    const config_3D_HeatmapOpacity = document.getElementById('config_3D_HeatmapOpacity');
    config_3D_HeatmapOpacity.oninput = function() {
        pointCloud.material.opacity = (this.value/100.0);
        pointCloud.material.needsUpdate = true;
    }
}


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
    const hideZeros = !document.getElementById('config_ShowZeros').checked;
    sos_coeffs.forEach((textElement, index) => {
        // Check if text is '0' and show zeros is disabled
        if (textElement.textContent === '0' && hideZeros) {
            textElement.style.display = 'none';
            return;
        }

        const pos = positions[index];
        // console.log(pos);

        // Create a ray from the camera to the point

        // Project 3D point to 2D screen
        const vector = new THREE.Vector3(pos.x,pos.y,pos.z).project(camera);
        const x = (vector.x * widthHalf + widthHalf - 6);
        const y = (-(vector.y * heightHalf) + heightHalf - 10);
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
        coeff.classList.add('coeff-label');
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

function renderHeatmap3D(heatmap){
    renderHeatmap(null);
    const pointCloud = _3d_vis.pointCloud;
    const color = pointCloud.geometry.attributes.color;
    if ((heatmap === undefined)||(heatmap === null)){
        // const degree = _3d_vis.heatmap_size_3d;
        // const num = (degree+1)*(degree+2)*(degree+3)/6;
        // for (let i=0;i<num;++i){
        //     color.setXYZ(i, 255, 255, 255);
        // }
        // color.needsUpdate = true;
        pointCloud.visible = false;
        return;
    }
    // _generateMonoms(4, _3d_vis.heatmap_size_3d).forEach((monom, i) => {
    //     const heatValue = heatmap[i];
    //     const color = new THREE.Color().setRGB(heatValue[0]/255, heatValue[1]/255, heatValue[2]/255);
    //     _3d_vis.pointCloud.geometry.attributes.color.setXYZ(i, color.r, color.g, color.b);
    // });
    // console.log(heatmap.length, color.count);
    heatmap.forEach((heatValue, i) => {
        color.setXYZ(i, heatValue[0]/255, heatValue[1]/255, heatValue[2]/255);
    });
    color.needsUpdate = true;
}


function toggleShowZeros(show) {
    // Toggle visibility of coefficient labels that have zero value
    sos_coeffs.forEach(coeff => {
        if (coeff.textContent === '0') {
            coeff.style.display = show ? 'block' : 'none';
        }
    });
}


function _init3DWebGLCheck(){
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') || canvas.getContext('webgl2');
    if (!gl) {
        return false;
    } else {
        return true;
    }
}

function _init3DVisualizationFallback(){
    // When WebGL is not supported
    const coeffs_view = VIEW3DPARENT;
    _3d_vis.scene = {};
    _3d_vis.camera = {};
    _3d_vis.renderer = {};
    _3d_vis.controls = {};
    _3d_vis.raycaster = {};
    _3d_vis.tetrahedron = coeffs_view.appendChild(document.createElement('div'));
    _3d_vis.tetrahedron.innerHTML = 
        'WebGL is not supported in your browser.<br>' +
        'Please visit <a href="https://get.webgl.org/" target="_blank">https://get.webgl.org/</a> for more information.';
    _3d_vis.tetrahedron.style.color = 'black';
    _3d_vis.tetrahedron.style.position = 'absolute';
    _3d_vis.tetrahedron.style.top = '15px';
    _3d_vis.tetrahedron.style.left = '15px';
    _3d_vis.tetrahedron.style.display = 'none';
    _3d_vis.positions = [];
    // _3d_vis.frustum = frustum;
    _3d_vis.pointCloud = {};
    _3d_vis.dashlines = null;
    _3d_vis.cameraViewProjectionMatrix = {};
    _3d_vis.tetrav = [];
}

if (_init3DWebGLCheck()){
    init3DVisualization();
}else{
    _init3DVisualizationFallback();
    updateTextPositions = () => {};
    renderCoeffs3D = () => {_3d_vis.in3d = true;};
    renderHeatmap3D = () => {renderHeatmap(null);};
    console.warn('WebGL is not supported in your browser.\nPlease visit https://get.webgl.org/ for more information.');
    console.log('WebGL is not supported in your browser.\nPlease visit https://get.webgl.org/ for more information.');
}