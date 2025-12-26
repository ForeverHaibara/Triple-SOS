
// Settings functionality

// Settings definition dictionary - defines the structure and properties of all settings
var settingsDefinition = {children: {
    parser: {
        name: "Parser Configs",
        id: "parser-configs",
        children: {
            // parser: {
            //     name: "Parser",
            //     description: "Choose the parser to use",
            //     type: "select",
            //     defaultValue: "pl",
            //     options: ["pl", "sympify"],
            //     settingPath: "parser.parser"
            // },
            cyclicSumFunc: {
                name: "Cyclic Sum Function",
                description: "Specify the syntax for cyclic sums",
                type: "text",
                defaultValue: "s",
                settingPath: "parser.cyclicSumFunc"
            },
            cyclicProdFunc: {
                name: "Cyclic Product Function",
                description: "Specify the syntax for cyclic products",
                type: "text",
                defaultValue: "p",
                settingPath: "parser.cyclicProdFunc"
            },
            preservePatterns: {
                name: "Preserve Patterns",
                description: "Specify patterns to preserve during parsing",
                type: "text",
                defaultValue: "sqrt",
                settingPath: "parser.preservePatterns"
            },
            lowerCase: {
                name: "Lower Case",
                description: "Convert the text to lower case",
                type: "checkbox",
                defaultValue: true,
                settingPath: "parser.lowerCase"
            },
            // convertLatex: {
            //     name: "Convert LaTeX",
            //     description: "Convert LaTeX expressions including \\frac, etc. This is experimental.",
            //     type: "checkbox",
            //     defaultValue: true,
            //     settingPath: "parser.convertLatex"
            // },
            scientificNotation: {
                name: "Scientific Notation",
                description: "Interpret numbers in scientific notation",
                type: "checkbox",
                defaultValue: false,
                settingPath: "parser.scientificNotation"
            },
            standardizeText: {
                name: "Standardize Text",
                id: "standardize-text",
                children: {
                    omitMul: {
                        name: "Omit Mul",
                        description: "Omit multiplication signs in output",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "parser.standardizeText.omitMul"
                    },
                    omitPow: {
                        name: "Omit Pow",
                        description: "Omit power signs in output",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "parser.standardizeText.omitPow"
                    }
                }
            }
        }
    },
    sos: {
        name: "SOS Configs",
        id: "sos-configs",
        children: {
            timeLimit: {
                name: "Time Limit",
                description: "Set the time limit for SOS solving in seconds. This will also be limited by the backend.",
                type: "number",
                defaultValue: 300.0,
                min: 0.0,
                // step: 1.0,
                settingPath: "sos.timeLimit"
            },
            structuralSOS: {
                name: "Structural SOS",
                id: "structural-sos",
                children: {
                    useStructuralSOS: {
                        name: "Use Structural SOS",
                        description: "Enable Structural SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.structuralSOS.useStructuralSOS",
                        originalCheckboxId: "setting_method_use_StructuralSOS"
                    }
                }
            },
            linearSOS: {
                name: "Linear SOS",
                id: "linear-sos",
                children: {
                    useLinearSOS: {
                        name: "Use Linear SOS",
                        description: "Enable Linear SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.linearSOS.useLinearSOS",
                        originalCheckboxId: "setting_method_use_LinearSOS"
                    },
                    liftDegreeLimit: {
                        name: "Lift Degree Limit",
                        description: "Set the maximum lifting degree for Linear SOS",
                        type: "number",
                        defaultValue: 4,
                        min: 0,
                        step: 1,
                        settingPath: "sos.linearSOS.liftDegreeLimit"
                    },
                    basisLimit: {
                        name: "Basis Limit",
                        description: "Set the maximum number of basis vectors for Linear SOS",
                        type: "number",
                        defaultValue: 15000,
                        min: 0,
                        step: 1,
                        settingPath: "sos.linearSOS.basisLimit"
                    },
                    quadDiffOrder: {
                        name: "Quadratic Difference Order",
                        description: "Set the order of quadratic difference for Linear SOS (should be even)",
                        type: "number",
                        defaultValue: 8,
                        min: 0,
                        step: 2,
                        settingPath: "sos.linearSOS.quadDiffOrder"
                    },
                    augmentTangents: {
                        name: "Augment Tangents",
                        description: "Whether to use heuristics to augment tangents in Linear SOS",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.linearSOS.augmentTangents"
                    },
                }
            },
            sdpSOS: {
                name: "SDP SOS",
                id: "sdp-sos",
                children: {
                    useSDPSOS: {
                        name: "Use SDP SOS",
                        description: "Enable SDP SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.sdpSOS.useSDPSOS",
                        originalCheckboxId: "setting_method_use_SDPSOS"
                    },
                    allowNumer: {
                        name: "Allow Numerical Solution",
                        description: "Allow numerical solution in SDP SOS",
                        type: "checkbox",
                        defaultValue: false,
                        settingPath: "sos.sdpSOS.allowNumer"
                    }
                }
            },
            symmetricSOS: {
                name: "Symmetric SOS",
                id: "symmetric-sos",
                children: {
                    useSymmetricSOS: {
                        name: "Use Symmetric SOS",
                        description: "Enable Symmetric SOS method",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.symmetricSOS.useSymmetricSOS",
                        originalCheckboxId: "setting_method_use_SymmetricSOS"
                    }
                }
            },
            pivoting: {
                name: "Pivoting",
                id: "pivoting",
                children: {
                    usePivoting: {
                        name: "Use Pivoting",
                        description: "Enable pivoting",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "sos.pivoting.usePivoting"
                    }
                }
            }
        }
    },
    result: {
        name: "Result Configs",
        id: "result-configs",
        children: {
            rewrite_symmetry: {
                name: "Rewrite Symmetry",
                description: "Whether to rewrite all symmetry groups to the input symmetry group.",
                type: "checkbox",
                defaultValue: true,
                settingPath: "result.rewrite_symmetry"
            },
            latex: {
                name: "LaTeX",
                id: "result-latex",
                children: {
                    together: {
                        name: "LaTeX Together",
                        description: "Whether to apply 'together' to the solution before conversion to LaTeX.",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "result.latex.together"
                    },
                    cancel: {
                        name: "LaTeX Cancel",
                        description: "Whether to apply 'cancel' to the solution before conversion to LaTeX.",
                        type: "checkbox",
                        defaultValue: true,
                        settingPath: "result.latex.cancel"
                    },
                    maxTermsAligned: {
                        name: "Max Terms Aligned",
                        description: "The number of terms to trigger line break in the LaTeX output.",
                        type: "number",
                        defaultValue: 2,
                        min: 0,
                        step: 1,
                        settingPath: "result.latex.maxTermsAligned"
                    },
                    maxLenAligned: {
                        name: "Max Length Aligned",
                        description: "The length of a line to trigger line break in the LaTeX output.",
                        type: "number",
                        defaultValue: 160,
                        min: 0,
                        step: 1,
                        settingPath: "result.latex.maxLenAligned"
                    },
                    maxLineLenInAligned: {
                        name: "Max Line Length in Aligned",
                        description: "The length of a line to trigger line break in the aligned environment in the LaTeX output.",
                        type: "number",
                        defaultValue: 100,
                        min: 0,
                        step: 1,
                        settingPath: "result.latex.maxLineLenInAligned"
                    }
                }
            },
            text: {
                name: "Plain Text",
                id: "result-text",
                children: {
                    together: {
                        name: "Text Together",
                        description: "Whether to apply 'together' to the solution before conversion to plain text.",
                        type: "checkbox",
                        defaultValue: false,
                        settingPath: "result.txt.together"
                    },
                    cancel: {
                        name: "Text Cancel",
                        description: "Whether to apply 'cancel' to the solution before conversion to plain text.",
                        type: "checkbox",
                        defaultValue: false,
                        settingPath: "result.txt.cancel"
                    }
                }
            }
        }
    }
}};

// Settings data structure with default values - initialized from settingsDefinition
var settings = {
};

// Helper function to generate a unique ID for settings controls
function generateControlId(settingPath) {
    return 'setting_' + settingPath.replace(/\./g, '_');
}

// Generate checkbox control HTML
function createCheckboxControl(setting) {
    var controlId = generateControlId(setting.settingPath);
    return '<div class="settings_item_control">' +
           '<input type="checkbox" id="' + controlId + '" class="settings_checkbox" ' + (setting.defaultValue ? 'checked' : '') + '>' +
           '<label for="' + controlId + '">Enable</label>' +
           '</div>';
}

// Generate text control HTML
function createTextControl(setting) {
    var controlId = generateControlId(setting.settingPath);
    return '<div class="settings_item_control">' +
           '<input type="text" id="' + controlId + '" class="settings_textbox" value="' + setting.defaultValue + '">' +
           '</div>';
}

// Generate dropdown menu HTML
function createSelectControl(setting) {
    var controlId = generateControlId(setting.settingPath);
    var optionsHtml = '';
    
    // Generate options for dropdown
    for (var i = 0; i < setting.options.length; i++) {
        var option = setting.options[i];
        var selected = (option === setting.defaultValue) ? 'selected' : '';
        optionsHtml += '<option value="' + option + '" ' + selected + '>' + option + '</option>';
    }
    
    return '<div class="settings_item_control">' +
           '<select id="' + controlId + '" class="settings_textbox">' +
           optionsHtml +
           '</select>' +
           '</div>';
}

// Generate number input HTML
function createNumberControl(setting) {
    var controlId = generateControlId(setting.settingPath);
    var minAttr = (setting.min !== undefined) ? ' min="' + setting.min + '"' : '';
    var maxAttr = (setting.max !== undefined) ? ' max="' + setting.max + '"' : '';
    var stepAttr = (setting.step !== undefined) ? ' step="' + setting.step + '"' : '';
    
    return '<div class="settings_item_control">' +
           '<input type="number" id="' + controlId + '" class="settings_textbox"' +
           ' value="' + setting.defaultValue + '"' + minAttr + maxAttr + stepAttr + '>' +
           '</div>';
}

// Generate HTML for a single setting item
function createSettingItem(setting) {
    var controlHtml = '';
    switch (setting.type) {
        case 'checkbox':
            controlHtml = createCheckboxControl(setting);
            break;
        case 'text':
            controlHtml = createTextControl(setting);
            break;
        case 'select':
            controlHtml = createSelectControl(setting);
            break;
        case 'number':
            controlHtml = createNumberControl(setting);
            break;
        // Add more control types here as needed
    }
    
    return '<div class="settings_item">' +
           '<div class="settings_item_name">' + setting.name + '</div>' +
           '<div class="settings_item_description">' + setting.description + '</div>' +
           controlHtml +
           '</div>';
}

// Recursively generate settings content from definition
function generateSettingsContent(parentNode, definition, level) {
    var html = '';

    // Determine if this is a section, subsection, or setting item
    if (definition.children) {
        // Create section or subsection header
        if (level >= 0){
            var headerTag = level === 0 ? 'h2' : 'h3';
            var headerClass = level === 0 ? 'settings_section_title' : 'settings_subsection_title';

            html += '<div id="settings_section_' + definition.id + '" class="settings_section">' +
                    '<' + headerTag + ' class="' + headerClass + '">' + definition.name + '</' + headerTag + '>';
        }
        // Recursively process children
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                html += generateSettingsContent(null, definition.children[key], level + 1);
            }
        }

        html += '</div>';
    } else {
        // This is a setting item
        html += createSettingItem(definition);
    }

    if (parentNode) {
        parentNode.innerHTML += html;
    }
    return html;
}

// Recursively generate settings menu from definition
function generateSettingsMenu(parentNode, definition, level) {
    var html = '';

    // Create menu item for sections and subsections
    if (definition.id) {
        var menuClass = level === 0 ? 'settings_menu_item' : 'settings_menu_item subitem';
        html += '<button class="' + menuClass + '" onclick="scrollToSection(\'settings_section_' + definition.id + '\')">' + definition.name + '</button>';
    }

    // Recursively process children
    if (definition.children) {
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                html += generateSettingsMenu(null, definition.children[key], level + 1);
            }
        }
    }

    if (parentNode) {
        parentNode.innerHTML += html;
    }
    return html;
}

// Initialize all settings from the definition
function initializeSettings() {
    // Initialize settings content
    var contentContainer = document.getElementById('settings_content');
    if (contentContainer) {
        contentContainer.innerHTML = '';
        generateSettingsContent(contentContainer, settingsDefinition, -1);
    }

    // Initialize settings menu
    var menuContainer = document.getElementById('settings_sidebar');
    if (menuContainer) {
        menuContainer.innerHTML = '';
        generateSettingsMenu(menuContainer, settingsDefinition, -1);
    }

    // Set initial values for all settings
    setInitialSettingsValues(settingsDefinition);

    // Setup event listeners for dynamic elements
    setupEventListenersRecursive(settingsDefinition);
}

// Set initial values for all settings
function setInitialSettingsValues(definition) {
    // If this is a section with children, recursively process them
    if (definition.children) {
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                setInitialSettingsValues(definition.children[key]);
            }
        }
    } else if (definition.type) {
        // This is a setting item
        var controlId = generateControlId(definition.settingPath);
        var element = document.getElementById(controlId);

        if (element) {
            // var value = getSettingValue(definition.settingPath);
            // if (definition.type === 'checkbox') {
            //     element.checked = value;
            // } else if (definition.type === 'text') {
            //     element.value = value;
            // }
            updateSetting(definition.settingPath, definition.defaultValue);
        }
    }
}

// Get setting value from the settings object
function getSettingValue(settingPath) {
    var pathParts = settingPath.split('.');
    var value = settings;

    for (var i = 0; i < pathParts.length; i++) {
        if (value && typeof value === 'object' && value.hasOwnProperty(pathParts[i])) {
            value = value[pathParts[i]];
        } else {
            return null;
        }
    }

    return value;
}

// Update setting value in the settings object
function updateSetting(settingPath, value) {
    var pathParts = settingPath.split('.');
    var obj = settings;

    // Navigate to the parent object
    for (var i = 0; i < pathParts.length - 1; i++) {
        if (obj && typeof obj === 'object' && obj.hasOwnProperty(pathParts[i])) {
            obj = obj[pathParts[i]];
        } else {
            // return; // Path doesn't exist
            // create the missing path
            obj[pathParts[i]] = {};
            obj = obj[pathParts[i]];
        }
    }

    // Update the value
    var lastKey = pathParts[pathParts.length - 1];
    obj[lastKey] = value;

    // Log the change for debugging (future: add save functionality here)
    // console.log('Setting updated:', settingPath, '=', value);
}


// Recursively setup event listeners
function setupEventListenersRecursive(definition) {
    if (definition.children) {
        // Process children recursively
        for (var key in definition.children) {
            if (definition.children.hasOwnProperty(key)) {
                setupEventListenersRecursive(definition.children[key]);
            }
        }
    } else if (definition.type) {
        // This is a setting item, setup event listener
                var controlId = generateControlId(definition.settingPath);
                var element = document.getElementById(controlId);
                
                if (element) {
                    if (definition.type === 'checkbox') {
                        // Checkbox change event
                        element.addEventListener('change', function() {
                            var value = this.checked;
                            updateSetting(definition.settingPath, value);
                            
                            // Sync with original checkbox if specified
                            if (definition.originalCheckboxId) {
                                var originalCheckbox = document.getElementById(definition.originalCheckboxId);
                                if (originalCheckbox) {
                                    originalCheckbox.checked = value;
                                }
                            }
                        });
                        
                        // Sync original checkbox changes to this one
                        if (definition.originalCheckboxId) {
                            var originalCheckbox = document.getElementById(definition.originalCheckboxId);
                            if (originalCheckbox) {
                                originalCheckbox.addEventListener('change', function() {
                                    element.checked = this.checked;
                                    updateSetting(definition.settingPath, this.checked);
                                });
                            }
                        }
                    } else if (definition.type === 'text') {
                        // Text input event
                        element.addEventListener('change', function() {
                            updateSetting(definition.settingPath, this.value);
                        });
                    } else if (definition.type === 'select') {
                        // Dropdown change event
                        element.addEventListener('change', function() {
                            updateSetting(definition.settingPath, this.value);
                        });
                    } else if (definition.type === 'number') {
                        // Number input event
                        element.addEventListener('change', function() {
                            // Convert to number if possible, otherwise use default
                            var value = parseFloat(this.value);
                            const thisvalue = value;
                            if (isNaN(value)) {
                                value = definition.defaultValue;
                            }
                            if (definition.step) {
                                value = Math.round(value / definition.step) * definition.step;
                            }
                            if (value < definition.min) {
                                value = definition.min;
                            }
                            if (value > definition.max) {
                                value = definition.max;
                            }
                            if (thisvalue !== value) {
                                this.value = value;
                            }
                            updateSetting(definition.settingPath, value);
                        });
                    }
                }
    }
}

// Open settings modal
function openSettingsModal() {
    document.getElementById('shadow').hidden = false;
    document.getElementById('settings_modal').hidden = false;
}

// Close settings modal
function closeSettingsModal() {
    document.getElementById('shadow').hidden = true;
    document.getElementById('settings_modal').hidden = true;
}

// Scroll to settings section
function scrollToSection(sectionId) {
    var section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Event listeners for modal controls
function setupModalEventListeners() {
    // Open button
    var openButton = document.getElementById('open_settings_btn');
    if (openButton) {
        openButton.addEventListener('click', openSettingsModal);
    }
    
    // Close button
    var closeButton = document.getElementById('settings_close');
    if (closeButton) {
        closeButton.addEventListener('click', closeSettingsModal);
    }
    
    // Click outside to close
    var shadow = document.getElementById('shadow');
    if (shadow) {
        shadow.addEventListener('click', function(e) {
            if (e.target.id === 'shadow') {
                closeSettingsModal();
            }
        });
    }

    // Escape key to close
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && !document.getElementById('settings_modal').hidden) {
            closeSettingsModal();
        }
    });
}


// Overwrite the default configs with the user configs
function getParserConfigs(parser_configs = {}) {
    const parse_undefined = (x, y) => {return typeof(x) === 'undefined'? y: x;}
    
    parser_configs.omit_mul = parse_undefined(
        parser_configs.omit_mul, settings.parser.standardizeText.omitMul);
    parser_configs.omit_pow = parse_undefined(
        parser_configs.omit_pow, settings.parser.standardizeText.omitPow);

    const parser_attrs = {
        // 'parser': 'parser',
        'lowerCase': 'lowercase',
        'cyclicSumFunc': 'cyclic_sum_func',
        'cyclicProdFunc': 'cyclic_prod_func',
        'preservePatterns': 'preserve_patterns',
        'scientificNotation': 'scientific_notation',
    };
    for (const [key, value] of Object.entries(parser_attrs)){
        parser_configs[value] = parse_undefined(parser_configs[value], settings.parser[key]);
    }
    return parser_configs;
}


function getSOSConfigs() {
    const sos = settings.sos;
    let methods = {
        StructuralSOS: sos.structuralSOS.useStructuralSOS,
        LinearSOS: sos.linearSOS.useLinearSOS,
        SymmetricSOS: sos.symmetricSOS.useSymmetricSOS,
        SDPSOS: sos.sdpSOS.useSDPSOS,
        Pivoting: sos.pivoting.usePivoting,
        Reparametrization: true,
    };
    methods = Object.entries(methods).filter(([key, value]) => value).map(([key, value]) => key);
    const configs = {
        StructuralSOS:{
            // real: document.getElementById("setting_method_StructuralSOS_real").checked,
        },
        LinearSOS: {
            lift_degree_limit: sos.linearSOS.liftDegreeLimit,
            basis_limit: sos.linearSOS.basisLimit,
            quad_diff_order: sos.linearSOS.quadDiffOrder,
            augment_tangents: sos.linearSOS.augmentTangents,
        },
        SDPSOS: {
            allow_numer: sos.sdpSOS.allowNumer,
            // decompose_method: document.getElementById("setting_method_SDPSOS_reduced_decompose").checked?'reduce':'raw',
        }
    }
    return {
        time_limit: sos.timeLimit,
        methods: methods,
        configs: configs,
    }
}


function getResultConfigs() {
    return {
        rewrite_symmetry: settings.result.rewrite_symmetry,
        to_string_configs: {
            latex: {
                together: settings.result.latex.together,
                cancel: settings.result.latex.cancel,
            },
            txt: {
                together: settings.result.txt.together,
                cancel: settings.result.txt.cancel,
            },
            formatted: {
            // use the same settings as txt
                together: settings.result.txt.together,
                cancel: settings.result.txt.cancel,
            },
        }
    }
}

// Initialize settings functionality when page loads
window.addEventListener('DOMContentLoaded', function() {
    initializeSettings();
    setupModalEventListeners();
    // initializeSettings will be called when modal opens
});
