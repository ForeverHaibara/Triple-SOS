// latex_auto_linebreak.js
// Converted from the original Python file to run in browser (ES6).
// Provides functions:
//  - getAdditiveTerms(latexStr)
//  - wrapWithAlignedIfNeeded(latexStr, maxTermsTrigger, maxLenTrigger, maxLineLen)
//  - recursiveLatexAutoLinebreak(latexStr, maxTermsAligned, maxLenAligned, maxLineLenInAligned)
// Example usage at the bottom prints results to console.

// -------------------- Constants / Helpers --------------------
const DELIM_PAIRS = {
    "\\left(": "\\right)",
    "\\left[": "\\right]",
    "\\left{": "\\right}",
    "\\left|": "\\right|",
    "\\left.": "\\right.",
    "(": ")",
    "[": "]",
};

const OPENER_REGEX_PATTERNS = [
    /\\left\(/, /\\left\[/, /\\left\{/, /\\left\|/, /\\left\./,
    /\\frac/,
    /\(/, /\[/
];

const OPENING_BRACKETS_FOR_DEPTH = ['(', '[', '{'];
const CLOSING_BRACKETS_FOR_DEPTH = [')', ']', '}'];

// Keep the additive-terms left delimiters in a deterministic array order
const ADDITIVE_TERMS_LEFT_DELIMS_ARR = [
    ["\\left(", "\\right)"],
    ["\\left[", "\\right]"],
    ["\\left{", "\\right}"],
    ["\\left|", "\\right|"],
    ["\\left.", "\\right."]
];

function _startsWithAt(s, sub, pos) {
    return s.substr(pos, sub.length) === sub;
}

// Attempt to extract one LaTeX 'argument' starting at startPos.
// Mirrors Python semantics:
//  - If it starts with '{', return the content inside matching braces (not including outer braces) and position after closing brace.
//  - If it starts with '\something' return that token (like \alpha or \sin or single-char escape) and new position.
//  - Otherwise return the single character and new position (startPos+1).
function _getLatexArgumentContent(text, startPos) {
    const n = text.length;
    if (startPos >= n) return { content: "", newPos: startPos };

    const ch = text[startPos];

    if (ch === "{") {
        let braceDepth = 0;
        for (let i = startPos; i < n; ++i) {
            if (text[i] === "{" && (i === 0 || text[i - 1] !== "\\")) {
                braceDepth += 1;
            } else if (text[i] === "}" && (i === 0 || text[i - 1] !== "\\")) {
                braceDepth -= 1;
                if (braceDepth === 0) {
                    return { content: text.slice(startPos + 1, i), newPos: i + 1 };
                }
            }
        }
        throw new Error(`Unmatched brace starting at ${startPos} in '${text}'`);
    } else if (ch === "\\") {
        const tail = text.slice(startPos);
        // Try to match a multi-letter command optionally followed by space and * or !
        const multiCmd = tail.match(/^\\[a-zA-Z]+(?:\s*[*!]?)?/);
        if (multiCmd) {
            const token = multiCmd[0];
            return { content: token, newPos: startPos + token.length };
        }
        // Fallback: single escaped char like \{
        const single = tail.match(/^\\./);
        if (single) {
            const token = single[0];
            return { content: token, newPos: startPos + token.length };
        }
        // Nothing matched, return the slash itself
        return { content: text[startPos], newPos: startPos + 1 };
    }
    return { content: text[startPos], newPos: startPos + 1 };
}

// -------------------- Part 1: Identify Additive Expressions --------------------
function getAdditiveTerms(latexStr) {
    if (typeof latexStr !== "string") return null;
    const terms = [];
    let buffer = "";
    const depthStack = [];
    const nStr = latexStr.length;
    let idx = 0;

    while (idx < nStr) {
        // check left delimiters (the special \left... forms)
        let foundLeftDelim = false;
        for (const [lDelimStr, rDelimStr] of ADDITIVE_TERMS_LEFT_DELIMS_ARR) {
            if (_startsWithAt(latexStr, lDelimStr, idx)) {
                depthStack.push(rDelimStr);
                buffer += latexStr.substr(idx, lDelimStr.length);
                idx += lDelimStr.length;
                foundLeftDelim = true;
                break;
            }
        }
        if (foundLeftDelim) continue;

        // check right delim for the last left \left... if any
        let foundRightDelim = false;
        if (depthStack.length > 0) {
            const expectedCloser = depthStack[depthStack.length - 1];
            if (expectedCloser.startsWith("\\right") && _startsWithAt(latexStr, expectedCloser, idx)) {
                depthStack.pop();
                buffer += latexStr.substr(idx, expectedCloser.length);
                idx += expectedCloser.length;
                foundRightDelim = true;
            }
        }
        if (foundRightDelim) continue;

        const char = latexStr[idx];

        if (OPENING_BRACKETS_FOR_DEPTH.includes(char)) {
            if (char === '(') depthStack.push(')');
            else if (char === '[') depthStack.push(']');
            else if (char === '{') depthStack.push('}');
            buffer += char;
            idx += 1;
        } else if (CLOSING_BRACKETS_FOR_DEPTH.includes(char)) {
            if (depthStack.length && depthStack[depthStack.length - 1] === char) depthStack.pop();
            buffer += char;
            idx += 1;
        } else if (_startsWithAt(latexStr, "\\frac", idx) && depthStack.length === 0) {
            // Note: original Python appended only the single char at idx in this case.
            buffer += char;
            idx += 1;
        } else if ((char === '+' || char === '-') && depthStack.length === 0) {
            if (buffer) terms.push(buffer.trim());
            buffer = char;
            idx += 1;
        } else {
            buffer += char;
            idx += 1;
        }
    }

    if (buffer) terms.push(buffer.trim());

    let finalTerms = [];
    if (!terms || terms.length === 0) return null;

    if (terms[0] === "" && terms.length > 1) {
        const firstTermCandidate = terms[1];
        if (firstTermCandidate && (firstTermCandidate.startsWith('+') || firstTermCandidate.startsWith('-'))) {
            finalTerms.push(firstTermCandidate);
            var startIdxForLoop = 2;
        } else {
            finalTerms.push(terms[0]);
            finalTerms.push(terms[1]);
            var startIdxForLoop = 2;
        }
    } else if (terms[0] !== "" && (terms[0].startsWith('+') || terms[0].startsWith('-'))) {
        finalTerms.push(terms[0]);
        var startIdxForLoop = 1;
    } else {
        finalTerms.push(terms[0]);
        var startIdxForLoop = 1;
    }

    for (let i = startIdxForLoop; i < terms.length; ++i) {
        finalTerms.push(terms[i]);
    }

    // Filter out empty terms except maybe keep the first original empty as Python logic did
    finalTerms = finalTerms.filter((t, idx) => (t || idx === 0));

    if (finalTerms.length && finalTerms[0] === "" && finalTerms.length > 1) {
        finalTerms.shift();
    }

    return finalTerms.length >= 2 ? finalTerms : null;
}

// -------------------- Part 2: Wrap with aligned if needed --------------------
const DEFAULT_MAX_TERMS_FOR_ALIGN = 2;
const DEFAULT_MAX_LEN_FOR_ALIGN = 160;
const DEFAULT_MAX_LINE_LEN_IN_ALIGNED = 100;

function wrapWithAlignedIfNeeded(
    latexStr,
    maxTermsTrigger = DEFAULT_MAX_TERMS_FOR_ALIGN,
    maxLenTrigger = DEFAULT_MAX_LEN_FOR_ALIGN,
    maxLineLen = DEFAULT_MAX_LINE_LEN_IN_ALIGNED
) {
    const terms = getAdditiveTerms(latexStr);

    let applyAlignedWrapper = false;
    if (terms) {
        if (terms.length > maxTermsTrigger || latexStr.length > maxLenTrigger) {
            applyAlignedWrapper = true;
        }
    }

    if (!applyAlignedWrapper) return latexStr;
    if (!terms) return latexStr; // safeguard

    const outputLinesAsTermLists = [];
    let currentLineTermGroup = [];

    // add first term to first group
    currentLineTermGroup.push(terms[0].trim());

    for (let i = 1; i < terms.length; ++i) {
        const termWithSign = terms[i].trim();

        // current line length as sum of lengths of terms
        const curLineLength = currentLineTermGroup.reduce((acc, t) => acc + t.length, 0);
        const newLineLength = curLineLength + termWithSign.length + (currentLineTermGroup.length ? 1 : 0);

        // compute the ratio heuristics used in Python
        const ratioOfCurrent = (curLineLength + 1) / (maxLineLen + 1) + (maxLineLen + 1) / (curLineLength + 1);
        const ratioOfExtended = (newLineLength + 1) / (maxLineLen + 1) + (maxLineLen + 1) / (newLineLength + 1);

        if ((curLineLength > maxLineLen || ratioOfExtended > ratioOfCurrent) && currentLineTermGroup.length > 0) {
            outputLinesAsTermLists.push(Array.from(currentLineTermGroup));
            currentLineTermGroup = [termWithSign];
        } else {
            currentLineTermGroup.push(termWithSign);
        }
    }

    if (currentLineTermGroup.length) {
        outputLinesAsTermLists.push(Array.from(currentLineTermGroup));
    }

    // Build aligned body parts
    const alignedBodyParts = [];
    for (let i = 0; i < outputLinesAsTermLists.length; ++i) {
        const termGroup = outputLinesAsTermLists[i];
        let lineStr = termGroup[0];
        for (let k = 1; k < termGroup.length; ++k) {
            lineStr += " " + termGroup[k];
        }
        lineStr = "& " + lineStr;
        alignedBodyParts.push(lineStr);
    }

    if (alignedBodyParts.length === 1) return latexStr;

    const alignedBody = alignedBodyParts.join("\\\\\n");
    return `\\begin{aligned}${alignedBody}\\end{aligned}`;
}

// -------------------- Part 3: Recursive Formatting --------------------
function _findNextOpenerInfo(latexStr, startPos, openerRegexPatterns) {
    const substring = latexStr.slice(startPos);
    let firstMatchPos = -1;
    let foundActualOpener = null;
    let foundOpenerRegex = null;

    for (const pattern of openerRegexPatterns) {
        // Use pattern without global state (the RegExp object may be reused)
        const re = new RegExp(pattern.source);
        const m = re.exec(substring);
        if (m) {
            const currentMatchPos = startPos + m.index;
            if (firstMatchPos === -1 || currentMatchPos < firstMatchPos) {
                firstMatchPos = currentMatchPos;
                foundActualOpener = m[0];
                foundOpenerRegex = pattern; // keep the original pattern object for reference
            }
        }
    }

    if (firstMatchPos === -1) return { matchPos: -1, actualOpenerText: null, openerRegex: null };
    return { matchPos: firstMatchPos, actualOpenerText: foundActualOpener, openerRegex: foundOpenerRegex };
}

function recursiveLatexAutoLinebreak(
    latexStr,
    maxTermsAligned = DEFAULT_MAX_TERMS_FOR_ALIGN,
    maxLenAligned = DEFAULT_MAX_LEN_FOR_ALIGN,
    maxLineLenInAligned = DEFAULT_MAX_LINE_LEN_IN_ALIGNED
) {
    const processedParts = [];
    let currentPos = 0;
    const n = latexStr.length;

    while (currentPos < n) {
        const { matchPos, actualOpenerText, openerRegex } = _findNextOpenerInfo(latexStr, currentPos, OPENER_REGEX_PATTERNS);

        if (matchPos === -1) {
            processedParts.push(latexStr.slice(currentPos));
            break;
        }

        if (matchPos > currentPos) {
            processedParts.push(latexStr.slice(currentPos, matchPos));
        }

        processedParts.push(actualOpenerText);
        currentPos = matchPos + actualOpenerText.length;

        if (actualOpenerText === "\\frac") {
            // numerator
            const { content: numContent, newPos: endPosNum } = _getLatexArgumentContent(latexStr, currentPos);
            const formattedNum = recursiveLatexAutoLinebreak(numContent, maxTermsAligned, maxLenAligned, maxLineLenInAligned);
            processedParts.push("{" + formattedNum + "}");
            currentPos = endPosNum;

            // denominator
            const { content: denContent, newPos: endPosDen } = _getLatexArgumentContent(latexStr, currentPos);
            const formattedDen = recursiveLatexAutoLinebreak(denContent, maxTermsAligned, maxLenAligned, maxLineLenInAligned);
            processedParts.push("{" + formattedDen + "}");
            currentPos = endPosDen;
        } else {
            const literalCloser = DELIM_PAIRS[actualOpenerText];
            if (!literalCloser) {
                // If no matching literal closer defined, continue scanning (replicates Python's continue)
                continue;
            }

            const contentStartIdx = currentPos;
            let level = 0;
            let searchIdx = contentStartIdx;
            let endDelimStartIdx = -1;

            while (searchIdx < n) {
                if (_startsWithAt(latexStr, actualOpenerText, searchIdx)) {
                    level += 1;
                    searchIdx += actualOpenerText.length;
                } else if (_startsWithAt(latexStr, literalCloser, searchIdx)) {
                    if (level === 0) {
                        endDelimStartIdx = searchIdx;
                        break;
                    }
                    level -= 1;
                    searchIdx += literalCloser.length;
                } else if (_startsWithAt(latexStr, "\\" + actualOpenerText, searchIdx)) {
                    // skip escaped opener
                    searchIdx += ("\\" + actualOpenerText).length;
                } else if (_startsWithAt(latexStr, "\\" + literalCloser, searchIdx)) {
                    // skip escaped closer
                    searchIdx += ("\\" + literalCloser).length;
                } else {
                    searchIdx += 1;
                }
            }

            if (endDelimStartIdx !== -1) {
                const content = latexStr.slice(contentStartIdx, endDelimStartIdx);
                const formattedContent = recursiveLatexAutoLinebreak(content, maxTermsAligned, maxLenAligned, maxLineLenInAligned);
                processedParts.push(formattedContent);
                processedParts.push(literalCloser);
                currentPos = endDelimStartIdx + literalCloser.length;
            } else {
                // unmatched case: just continue (same as Python 'pass')
            }
        }
    }

    const finalReconstructedStr = processedParts.join("");
    return wrapWithAlignedIfNeeded(finalReconstructedStr, maxTermsAligned, maxLenAligned, maxLineLenInAligned);
}

// -------------------- Example usage (prints to console) --------------------
function _exampleLatexAutoLinebreak() {
    console.log("--- Part 1: getAdditiveTerms (sanity check) ---");
    const s1 = "4a+2b-3c/5+(c-a)^2/2";
    console.log(`'${s1}' ->`, getAdditiveTerms(s1));
    const sLongSum = "a+b+c+d+e+f+g+h+i";
    console.log(`'${sLongSum}' ->`, getAdditiveTerms(sLongSum));

    console.log("\n--- Part 2: wrapWithAlignedIfNeeded (with new line length logic) ---");
    const exprToWrap = "termA+termB+termC+termD+termE+termF+termG";
    console.log(`\nInput: '${exprToWrap}'`);
    console.log("Wrapped (max_terms=2, max_len=20, max_line_len=15):");
    console.log(wrapWithAlignedIfNeeded(exprToWrap, 2, 20, 15));

    console.log(`\nInput: '${sLongSum}' (a+b+c+d+e+f+g+h+i)`);
    console.log("Wrapped (max_terms=2, max_len=5, max_line_len=10):");
    console.log(wrapWithAlignedIfNeeded(sLongSum, 2, 5, 10));

    console.log("\n--- Part 3: recursiveLatexAutoLinebreak (passing new param) ---");
    const longParenExpr = "f(x) = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r) + (s+t+u+v+w) - 2(x+y+z)";
    console.log(`\nInput: '${longParenExpr}'`);
    console.log("Recursively formatted (max_terms_aligned=2, max_len_aligned=30, max_line_len_in_aligned=20):");
    console.log(recursiveLatexAutoLinebreak(longParenExpr, 2, 30, 20));

    console.log("-".repeat(20));
    const nestedFracLongNum = "E = \\frac{X_1+X_2+X_3+X_4+X_5+X_6+X_7+X_8+X_9+X_{10}}{k+l+m} + Y";
    console.log(`\nInput: '${nestedFracLongNum}'`);
    console.log("Recursively formatted (max_terms_aligned=1, max_len_aligned=10, max_line_len_in_aligned=18):");
    console.log(recursiveLatexAutoLinebreak(nestedFracLongNum, 1, 10, 18));
}


// -------------------- Exports --------------------
// If used in module environment (e.g., bundler), export functions:
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
    module.exports = {
        getAdditiveTerms,
        wrapWithAlignedIfNeeded,
        recursiveLatexAutoLinebreak
    };
}
