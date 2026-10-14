// The HTML template is constant. Only props.value changes, so the graph node
// and its WebGL scene survive every frame instead of being remounted.
const graph = element.querySelector('.pose-graph');
let initialized = false;
let updating = false;
let pending = null;

async function updatePlot() {
    if (updating) return;
    updating = true;
    try {
        while (pending !== null) {
            const value = pending;
            pending = null;
            const figure = typeof value === 'string' ? JSON.parse(value) : value;
            if (!figure) continue;
            const layout = {...figure.layout, autosize: true};
            const config = {responsive: true, displaylogo: false};
            if (!initialized) {
                await window.Plotly.newPlot(graph, figure.data, layout, config);
                initialized = true;
            } else {
                // Stable uirevision preserves the camera inside this SAME scene.
                // Do not set or restore scene.camera during data updates.
                await window.Plotly.react(graph, figure.data, layout, config);
            }
        }
    } catch (error) {
        console.error('ArUco 3D plot update failed:', error);
    } finally {
        updating = false;
    }
}
function receive() {
    pending = props.value;
    updatePlot();
}
watch('value', receive);
const resizeObserver = new ResizeObserver(() => {
    if (initialized) window.Plotly.Plots.resize(graph);
});
resizeObserver.observe(element);
receive();
