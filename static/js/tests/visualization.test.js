// Run with: node --test static/js/tests/visualization.test.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const THREE = require('../three.min.js');

function frontend(webgl = true){
    const elements = new Map();
    function element(){
        return {
            style: {}, children: [], classList: {add: () => {}},
            clientWidth: 600, clientHeight: 500, value: '', checked: true,
            textContent: '',
            set innerText(value){ this.textContent = String(value); },
            set innerHTML(value){ this.textContent = String(value); },
            appendChild(child){ this.children.push(child); return child; },
            removeChild(child){ this.children.splice(this.children.indexOf(child), 1); },
            getBoundingClientRect: () => ({width: 600, height: 500}),
            setAttribute: () => {}, getContext: () => webgl ? {} : null,
        };
    }
    const context = vm.createContext({
        console, requestAnimationFrame: () => {},
        THREE: {...THREE, WebGLRenderer: class {
            constructor(){ this.domElement = element(); }
            setSize(){} render(){}
        }, OrbitControls: class {}},
        document: {createElement: element, getElementById(id){
            if (!elements.has(id)) elements.set(id, element());
            return elements.get(id);
        }},
        sos_coeffs: [], sos_poly: '', heatmap_gridsize: -1,
    });
    const run = code => vm.runInContext(code, context);
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../visualization.js'), 'utf8'), context);
    return {context, run, hull: points => {
        context.testPoints = points.map(point => new THREE.Vector3(...point));
        return run('_computeConvexHull3D(testPoints)');
    }, render(degree, values, mode = 'hull'){
        context.testValues = values;
        context.testDegree = degree;
        run(`document.getElementById('config_3D_Geometry').value = '${mode}';
            renderVisualization({n: testDegree, triangle: testValues, gens: 'abcd', txt: 'test'});`);
    }};
}

function checkSolidHull(hull, volume){
    const edgeCounts = new Map();
    let actualVolume = 0;
    for (const [i, j, k] of hull.faces){
        const [a, b, c] = [i, j, k].map(index => hull.points[index]);
        const normal = b.clone().sub(a).cross(c.clone().sub(a));
        assert.ok(normal.length() > 1e-10, 'faces must have positive area');
        normal.normalize();
        for (const point of hull.points){
            assert.ok(normal.dot(point.clone().sub(a)) < 1e-6, 'every face must support the entire point set');
        }
        actualVolume += a.dot(b.clone().cross(c))/6;
        for (const [u, v] of [[i, j], [j, k], [k, i]]){
            const key = [u, v].sort((x, y) => x-y).join(',');
            edgeCounts.set(key, (edgeCounts.get(key) || 0)+1);
        }
    }
    assert.ok(actualVolume > 0, 'hull must enclose positive volume');
    for (const count of edgeCounts.values()) assert.equal(count, 2, 'surface must be closed');
    if (volume !== undefined) assert.ok(Math.abs(actualVolume-volume) < 1e-6);
}

test('solid hulls enclose all points and omit coplanar wireframe diagonals', () => {
    const app = frontend();
    const cube = [];
    for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) cube.push([x, y, z]);
    const hull = app.hull(cube);
    checkSolidHull(hull, 8);
    // Collinear boundary segments may split an edge, but no face diagonal is drawn.
    for (const [i, j] of hull.edges){
        const a = hull.points[i], b = hull.points[j];
        assert.equal(['x', 'y', 'z'].filter(axis => Math.abs(a[axis]-b[axis]) > 1e-6).length, 1);
    }
    const tetrahedron = app.hull([[1,1,1], [-1,-1,1], [-1,1,-1], [1,-1,-1], [0,0,0]]);
    checkSolidHull(tetrahedron, 8/3);
    assert.equal(tetrahedron.edges.length, 6);
});

test('empty, coincident, collinear and coplanar supports have valid boundaries', () => {
    const app = frontend();
    assert.equal(app.hull([]).faces.length, 0);
    assert.equal(app.hull([[1,2,3], [1,2,3]]).points.length, 1);
    const line = app.hull([[2,3,4], [3,4,5], [0,1,2], [1,2,3]]);
    assert.equal(line.faces.length, 0);
    assert.equal(line.edges.length, 1);
    assert.ok(Math.abs(line.points[line.edges[0][0]].distanceTo(line.points[line.edges[0][1]]) - Math.sqrt(27)) < 1e-6);
    const plane = app.hull([[2,0,0], [2,1,0], [2,1,1], [2,0,1], [2,0.5,0.5], [2,0.5,0]]);
    assert.equal(plane.faces.length, 2);
    assert.equal(plane.edges.length, 4);
});

test('incremental hull remains closed for shuffled lattice points and random supports', () => {
    const app = frontend();
    let seed = 24601;
    const random = () => { seed = (1664525*seed + 1013904223) >>> 0; return seed/2**32; };
    const lattice = [];
    for (let i = 0; i <= 6; ++i) for (let j = 0; j <= 6-i; ++j) for (let k = 0; k <= 6-i-j; ++k) lattice.push([i/6, j/6, k/6]);
    for (let trial = 0; trial < 30; ++trial){
        for (let i = lattice.length-1; i > 0; --i){
            const j = Math.floor(random()*(i+1));
            [lattice[i], lattice[j]] = [lattice[j], lattice[i]];
        }
        checkSolidHull(app.hull(lattice), 1/6);
        checkSolidHull(app.hull(lattice.filter(() => random() < 0.35)));
        checkSolidHull(app.hull(Array.from({length: 40}, () => [random(), random(), random()])));
    }
});

test('mode changes update occlusion for an off-center hull and release old geometry', () => {
    const app = frontend();
    const monoms = app.run('_generateMonoms(4, 16)');
    const support = [[8,4,2,2], [4,8,2,2], [4,4,6,2], [4,4,2,6], [6,6,2,2], [4,4,4,4], [5,5,3,3]];
    const values = Array.from(monoms, monom => support.some(point => point.every((value, i) => value === monom[i])) ? '1' : '0');
    app.render(16, values, 'tetrahedron');
    const state = app.run('_3d_vis');
    const labelAt = monom => app.context.sos_coeffs[monoms.findIndex(point => point.every((value, i) => value === monom[i]))];
    const front = labelAt([6,6,2,2]);
    assert.equal(front.style.opacity, 0.2);
    let disposed = false, outlineDisposed = false, materialDisposed = false;
    state.tetrahedron.geometry.addEventListener('dispose', () => { disposed = true; });
    state.outline.geometry.addEventListener('dispose', () => { outlineDisposed = true; });
    state.outline.material.addEventListener('dispose', () => { materialDisposed = true; });
    app.run("document.getElementById('config_3D_Geometry').value = 'hull'; set3DGeometry();");
    assert.equal(disposed, true);
    assert.equal(outlineDisposed, true);
    assert.equal(materialDisposed, true);
    assert.equal(front.style.opacity, 1);
    assert.equal(labelAt([5,5,3,3]).style.opacity, 0.2);
    assert.equal(labelAt([4,4,4,4]).style.opacity, 0.2);
    state.camera.position.set(0, 0, -5);
    app.run('updateTextPositions()');
    assert.equal(labelAt([4,4,4,4]).style.opacity, 1);
    assert.equal(front.style.opacity, 0.2);
    app.run("document.getElementById('config_ShowZeros').checked = false; updateTextPositions();");
    assert.equal(labelAt([16,0,0,0]).style.display, 'none');
    assert.equal(state.hull.points.length, support.length);
    app.run("document.getElementById('config_3D_Geometry').value = 'tetrahedron'; set3DGeometry();");
    assert.equal(front.style.opacity, 0.2);
});

test('planar surfaces occlude labels behind them but keep their own labels opaque', () => {
    const app = frontend();
    const hull = app.hull([[1,-1,0], [3,-1,0], [3,1,0], [1,1,0], [2,0,0]]);
    app.context.testHull = hull;
    app.run(`_3d_vis.hull = testHull;
        document.getElementById('config_3D_Geometry').value = 'hull'; set3DGeometry();`);
    const mesh = app.run('_3d_vis.tetrahedron');
    const camera = new THREE.Vector3(2,0,5);
    const raycaster = new THREE.Raycaster(camera, new THREE.Vector3(0,0,-1));
    const hits = raycaster.intersectObject(mesh, false);
    assert.equal(hits[0].distance, 5);
    assert.ok(hits[0].distance < camera.distanceTo(new THREE.Vector3(2,0,-1)));
});

test('a monomial hull keeps a small pixel-sized marker when zooming or rotating', () => {
    const app = frontend();
    const monoms = app.run('_generateMonoms(4, 5)');
    for (const monom of [[5,0,0,0], [1,1,2,1]]){
        const index = monoms.findIndex(point => point.every((value, i) => value === monom[i]));
        app.render(5, Array.from(monoms, (_, i) => i === index ? '-7' : '0'));
        const state = app.run('_3d_vis');
        assert.equal(state.outline.geometry.attributes.position.count, 1);
        assert.ok(new THREE.Vector3().fromBufferAttribute(state.outline.geometry.attributes.position, 0)
            .distanceTo(state.positions[index]) < 1e-6);
        assert.equal(state.outline.material.size, 4);
        assert.equal(state.outline.material.sizeAttenuation, false);
        for (const camera of [[0,0,3], [0,0,8], [5,0,0], [0,0,-5]]){
            state.camera.position.set(...camera);
            state.camera.lookAt(0,0,0);
            app.run('updateTextPositions()');
            assert.equal(app.context.sos_coeffs[index].style.display, 'block');
            assert.equal(app.context.sos_coeffs[index].style.opacity, 1);
        }
    }
});

test('new data, constants, zero polynomials and 2D transitions preserve the selected mode', () => {
    const app = frontend();
    app.render(1, ['1','1','1','1']);
    app.render(1, ['0','1','0','0']);
    assert.equal(app.run('_3d_vis.hull.points.length'), 1);
    assert.equal(app.run('_3d_vis.outline.isPoints'), true);
    app.render(1, ['0','0','0','0']);
    assert.equal(app.run('_3d_vis.tetrahedron.geometry.index.count'), 0);
    app.render(0, ['7']);
    assert.equal(app.context.sos_coeffs.length, 1);
    assert.equal(app.context.sos_coeffs[0].style.opacity, 1);
    app.run("renderVisualization({n: 1, triangle: ['1','2','3'], gens: 'abc', txt: 'test'});");
    assert.equal(app.run('_3d_vis.tetrahedron.visible'), false);
    app.render(1, ['1','1','1','1']);
    assert.equal(app.run('_3d_vis.tetrahedron.visible'), true);
    assert.equal(app.run("document.getElementById('config_3D_Geometry').value"), 'hull');
});

test('geometry control remains harmless without WebGL', () => {
    const app = frontend(false);
    app.render(1, ['1','0','0','0']);
    app.run('set3DGeometry()');
    assert.equal(app.run('_3d_vis.tetrahedron.style.display'), 'block');
});
