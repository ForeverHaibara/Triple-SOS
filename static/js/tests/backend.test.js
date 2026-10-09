// Run with: node --test static/js/tests/backend.test.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function frontend(options = {}){
    const elements = new Map();
    const intervals = new Map();
    const sockets = [];
    const requests = [];
    const storage = options.storage || new Map();
    const scriptPath = path.join(__dirname, '..');
    let timerId = 0;
    function element(id){
        if (elements.has(id)) return elements.get(id);
        const classes = new Set();
        const listeners = new Map();
        const node = {
            value: '', textContent: '', hidden: false, disabled: false,
            classList: {
                contains: value => classes.has(value),
                toggle: (value, enabled) => enabled ? classes.add(value) : classes.delete(value),
            },
            addEventListener: (event, callback) => listeners.set(event, callback),
            fire: event => listeners.get(event)?.(),
            focus: () => {}, blur: () => {}, select: () => { node.selected = true; },
            querySelectorAll: () => [],
        };
        elements.set(id, node);
        return node;
    }
    const context = vm.createContext({
        URL, Date, setTimeout, console: {error: () => {}},
        host: options.host || 'http://127.0.0.1:5000',
        navigator: {userAgent: 'Windows', clipboard: options.clipboard},
        window: {location: {href: options.href || 'file:///D:/Triple-SOS/triples.html'}},
        document: {getElementById: element, querySelector: () => ({value: 'cyc'})},
        sessionStorage: {
            getItem: key => {
                if (options.storageBlocked) throw new Error('Storage denied');
                return storage.get(key);
            },
            setItem: (key, value) => {
                if (options.storageBlocked) throw new Error('Storage denied');
                storage.set(key, value);
            },
        },
        bootstrap: {Modal: class {
            constructor(node){ this.node = node; }
            show(){
                this.node.classList.toggle('show', true);
                this.node.fire('shown.bs.modal');
            }
            hide(){
                this.node.classList.toggle('show', false);
                this.node.fire('hide.bs.modal');
            }
        }},
        setInterval: callback => { intervals.set(++timerId, callback); return timerId; },
        clearInterval: id => intervals.delete(id),
        io: (host, settings) => {
            const handlers = new Map();
            const socket = {
                host, settings, connected: false, id: 'test-session',
                on: (event, callback) => handlers.set(event, callback),
                removeAllListeners: () => handlers.clear(),
                disconnect: () => { socket.closed = true; socket.connected = false; },
                fire: event => {
                    if (event === 'connect') socket.connected = true;
                    if (event === 'disconnect') socket.connected = false;
                    handlers.get(event)?.();
                },
            };
            sockets.push(socket);
            return socket;
        },
        axios: {isAxiosError: error => error.isAxiosError === true, post: (url, data) => {
            requests.push({url, data});
            return options.post ? options.post(url, data) : Promise.resolve({data: {}});
        }},
        sos_work: {num: 0}, sos_poly: '', sos_results: {},
        changeNumOfSOS: num => { context.sos_work.num = Math.max(0, num); },
        getParserConfigs: data => data || {}, getSOSConfigs: () => ({}), getResultConfigs: () => ({}),
        isSameHistoryRecord: () => false, history_data: [],
        updateHistoryData: () => {}, setHistoryByTimestamp: () => {}, renderVisualization: () => {},
    });
    element('input_poly').value = 'a2+b2+c2';
    element('setting_gens_input').value = 'abc';
    element('setting_perm_input').value = '[[1,2,0]]';
    for (const file of ['backend.js', 'socket.js']){
        vm.runInContext(fs.readFileSync(path.join(scriptPath, file), 'utf8'), context);
    }
    const html = fs.readFileSync(path.join(scriptPath, '../../triples.html'), 'utf8');
    const inline = Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/g));
    vm.runInContext(inline.at(-1)[1], context);
    return {context, elements, intervals, sockets, requests, storage, element,
        run: code => vm.runInContext(code, context)};
}

test('startup commands preserve local paths and quote shell metacharacters', () => {
    const page = frontend();
    assert.equal(page.run(`backendStartCommand('windows', "file:///C:/My%20SOS/O'Hara/triples.html")`),
        "Set-Location -LiteralPath 'C:\\My SOS\\O''Hara' -ErrorAction Stop; python .\\web_main.py");
    assert.equal(page.run(`backendStartCommand('posix', "file:///Users/Me/O'Hara%20%24SOS/triples.html")`),
        "cd -- '/Users/Me/O'\"'\"'Hara $SOS' && python3 web_main.py");
    assert.equal(page.run(`backendStartCommand('windows', 'file://server/share/SOS/triples.html')`),
        "Set-Location -LiteralPath '\\\\server\\share\\SOS' -ErrorAction Stop; python .\\web_main.py");
    assert.equal(page.run(`backendStartCommand('posix', 'https://example.com/')`), 'python3 web_main.py');
    assert.equal(page.run(`backendStartCommand('posix', 'file:///bad%0Apath/triples.html')`), 'python3 web_main.py');
    assert.equal(page.run(`backendStartCommand('posix', 'file:///bad%path/triples.html')`), 'python3 web_main.py');
});

test('concurrent requests wait for one connection and run only once', () => {
    const page = frontend();
    page.run('var calls = 0; connectSocket(() => calls++); connectSocket(() => calls++);');
    assert.equal(page.sockets.length, 1);
    assert.equal(page.run('calls'), 0);
    page.sockets[0].fire('connect');
    page.sockets[0].fire('connect');
    assert.equal(page.run('calls'), 2);
    page.run('connectSocket(() => calls++);');
    assert.equal(page.run('calls'), 3);
    assert.equal(page.sockets.length, 1);
});

test('failed connections discard stale callbacks and show only one dialog', () => {
    const page = frontend();
    page.run('var calls = 0; connectSocket(() => calls++); connectSocket(() => calls++);');
    page.sockets[0].fire('connect_error');
    assert.ok(page.sockets[0].closed);
    assert.equal(page.run('calls'), 0);
    assert.ok(page.element('backend_modal').classList.contains('show'));
    assert.equal(page.intervals.size, 1);
    // The dialog probes the backend silently; repeated errors must not multiply dialogs.
    page.sockets[1].fire('connect_error');
    assert.equal(page.intervals.size, 1);
    [...page.intervals.values()][0]();
    page.sockets[2].fire('connect');
    assert.equal(page.run('calls'), 0);
    assert.match(page.element('backend_message').textContent, /^Connected/);
    assert.ok(page.element('backend_message').classList.contains('text-success'));
    assert.ok(page.element('backend_message').classList.contains('fw-semibold'));
    assert.ok(!page.element('backend_message').classList.contains('text-secondary'));
    page.run('setBackendMessage("Still unable to connect.")');
    assert.ok(!page.element('backend_message').classList.contains('text-success'));
    assert.ok(!page.element('backend_message').classList.contains('fw-semibold'));
    assert.ok(page.element('backend_message').classList.contains('text-secondary'));
    page.run('backend_modal.hide()');
    assert.equal(page.intervals.size, 0);
    page.run('connectSocket(() => calls++)');
    assert.equal(page.run('calls'), 1);
});

test('prompt suppression resets on reload and manual help still works', () => {
    const page = frontend();
    page.run('showBackendPrompt()');
    page.element('backend_dismiss_session').fire('click');
    assert.equal(page.intervals.size, 0);
    page.run('showBackendPrompt()');
    assert.ok(!page.element('backend_modal').classList.contains('show'));
    page.element('backend_status').fire('click');
    assert.ok(page.element('backend_modal').classList.contains('show'));
    // Ignore preferences stored by older versions of the frontend.
    page.storage.set('triples_backend_prompt_disabled', 'true');
    const reloaded = frontend({storage: page.storage});
    reloaded.run('showBackendPrompt()');
    assert.ok(reloaded.element('backend_modal').classList.contains('show'));
});

test('storage and clipboard denial leave usable manual controls', async () => {
    const page = frontend({storageBlocked: true});
    await page.run('copyBackendCommand()');
    assert.ok(page.element('backend_command').selected);
    assert.match(page.element('backend_message').textContent, /copy the command/);
    page.element('backend_dismiss_session').fire('click');
    page.run('showBackendPrompt()');
    assert.ok(!page.element('backend_modal').classList.contains('show'));
});

test('remote backend failures offer connection help without local startup commands', () => {
    const page = frontend({host: 'https://example.com'});
    page.run('showBackendPrompt()');
    assert.equal(page.element('backend_title').textContent, 'Backend unavailable');
    assert.ok(page.element('backend_local_help').hidden);
    assert.ok(!page.element('backend_remote_help').hidden);
    page.run('backend_modal.hide(); handleBackendError({response: {status: 500}})');
    assert.ok(!page.element('backend_modal').classList.contains('show'));
    assert.match(page.element('backend_status').title, /HTTP 500/);
});

test('SOS counts start on dispatch and reset after errors or invalid input', async () => {
    const page = frontend({post: () => Promise.reject({response: {status: 500}})});
    page.run('SumOfSquares()');
    assert.equal(page.context.sos_work.num, 0);
    page.sockets[0].fire('connect');
    assert.equal(page.context.sos_work.num, 1);
    assert.equal(page.requests[0].data.sid, 'test-session');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(page.context.sos_work.num, 0);
    page.element('input_poly').value = '';
    page.run('SumOfSquares()');
    assert.equal(page.context.sos_work.num, 0);
    assert.equal(page.requests.length, 1);
});

test('preprocessing and coefficient export failures both offer startup help', async () => {
    for (const action of ['preprocessInput()', "PolyTools('latex_coeffs')"]){
        const page = frontend({post: () => Promise.reject({isAxiosError: true, message: 'Network Error'})});
        page.run(action);
        if (page.sockets.length) page.sockets[0].fire('connect');
        await new Promise(resolve => setImmediate(resolve));
        assert.ok(page.element('backend_modal').classList.contains('show'));
    }
});

test('disconnecting during SOS clears the counter and permits a fresh connection', () => {
    const page = frontend();
    page.run('connectSocket()');
    page.sockets[0].fire('connect');
    page.context.sos_work.num = 1;
    page.sockets[0].fire('disconnect');
    assert.equal(page.context.sos_work.num, 0);
    assert.ok(page.sockets[0].closed);
    assert.ok(page.element('backend_modal').classList.contains('show'));
    page.sockets[1].fire('connect');
    assert.ok(page.element('backend_status').classList.contains('text-success'));
});
