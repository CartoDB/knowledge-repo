---
authors:
- eschbacher
created_at: 2019-06-29 00:00:00
tags:
- cartoframes
- example
title: CARTOFrames Bikeshare Example with v1.0b2
tldr: Example of POSTGIS SQL, Enrichment with DO v1, Legends,  Grided maps, Publishing
  an App
updated_at: 2019-08-21 17:25:00.490307
---

```python
from cartoframes.__version__ import __version__
print(__version__)
# note: this is actually from `develop`: pip install -e git+https://github.com/cartodb/cartoframes.git@develop#egg=cartoframes
```
    1.0b2



```python
from cartoframes.auth import set_default_credentials, Credentials
from cartoframes.client import SQLClient
from cartoframes.data import Dataset
from cartoframes.viz import Map, Layer
import pandas as pd
import geopandas as gpd

creds = Credentials('eschbacher', 'x')

set_default_credentials(creds)

station_points = 'capital_bikeshare_stations_points_arlington'
usage_data = 'capitalbikeshare_tripdata_201907'

sql_client = SQLClient(creds)
```

```python
Dataset(usage_data).download(limit=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>start_date</th>
      <th>end_date</th>
      <th>start_station_number</th>
      <th>start_station</th>
      <th>end_station_number</th>
      <th>end_station</th>
      <th>bike_number</th>
      <th>member_type</th>
      <th>the_geom</th>
    </tr>
    <tr>
      <th>cartodb_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>443.0</td>
      <td>2019-07-01 04:11:52+00:00</td>
      <td>2019-07-01 04:19:16+00:00</td>
      <td>31603.0</td>
      <td>1st &amp; M St NE</td>
      <td>31620.0</td>
      <td>5th &amp; F St NW</td>
      <td>W24210</td>
      <td>Member</td>
      <td></td>
    </tr>
    <tr>
      <th>354</th>
      <td>500.0</td>
      <td>2019-07-01 06:22:18+00:00</td>
      <td>2019-07-01 06:30:39+00:00</td>
      <td>31244.0</td>
      <td>4th &amp; E St SW</td>
      <td>31652.0</td>
      <td>4th &amp; M St SE</td>
      <td>W24154</td>
      <td>Member</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
Dataset(station_points).download(limit=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>objectid</th>
      <th>gisid</th>
      <th>name</th>
      <th>the_geom</th>
    </tr>
    <tr>
      <th>cartodb_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>37.0</td>
      <td>31001.0</td>
      <td>18th &amp; Eads St</td>
      <td>0101000020E610000085093973704353C036DDEA41B96D...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>31014.0</td>
      <td>Lynn &amp; 19th St N</td>
      <td>0101000020E6100000712065E48A4453C078CEF79FE072...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from cartoframes.viz.helpers import size_continuous_layer
```

```python
from cartoframes.viz.helpers import color_continuous_layer
```
### Visualize the data


```python
sql = f'''
SELECT
    _ends.num_bike_dropoffs,
    _starts.num_bike_pickups,
    abs(_ends.num_bike_dropoffs - _starts.num_bike_pickups) as diff,
    CASE WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups > 0 THEN 1
         WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups = 0 THEN 0
         ELSE -1 END as diff_sign,
    _ends.num_bike_dropoffs - _starts.num_bike_pickups as diff_relative,
    _starts.station_id,
    row_number() OVER () as cartodb_id,
    ST_X(_starts.the_geom) as longitude,
    ST_Y(_starts.the_geom) as latitude,
    _starts.the_geom,
    ST_Transform(_starts.the_geom, 3857) as the_geom_webmercator
FROM
    (SELECT
      count(u.*) as num_bike_dropoffs,
      u.end_station_number::int as station_id,
      s.the_geom,
      s.cartodb_id
    FROM {station_points} as s
    JOIN {usage_data} as u
    ON u.end_station_number::int = s.gisid::int
    GROUP BY 2, 3, 4) as _ends
JOIN
    (SELECT
      count(u.*) as num_bike_pickups,
      u.start_station_number::int as station_id,
      s.the_geom,
      s.cartodb_id
    FROM {usage_data} as u
    JOIN {station_points} as s
    ON u.start_station_number::int = s.gisid::int
    GROUP BY 2, 3, 4) as _starts
ON _ends.station_id = _starts.station_id
'''
Map([
    size_continuous_layer(Dataset(sql), 'num_bike_pickups'),
#     color_continuous_layer(Dataset(sql), 'diff_relative')
    size_continuous_layer(Dataset(sql), 'num_bike_dropoffs', color='purple')  
])
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 632px;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>None</title>
  <meta name=&quot;description&quot; content=&quot;None&quot;>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
  <style>
  as-widget-header .as-widget-header__header {
    margin-bottom: 8px;
  }

  as-widget-header .as-widget-header__subheader {
    margin-bottom: 12px;
  }

  as-category-widget {
    max-height: 250px;
  }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <img id=&quot;map-image&quot; class=&quot;map-image&quot; alt='Static map image' />
  <as-responsive-content id=&quot;main-container&quot;>

    <main class=&quot;as-main&quot;>
      <div class=&quot;as-map-area&quot;>
        <div id=&quot;map&quot; class=&quot;map&quot;></div>


          <div class=&quot;as-map-panels&quot; data-name=&quot;Legends&quot;>
            <div class=&quot;as-panel as-panel--left as-panel--top&quot;>


<div class=&quot;as-panel__element&quot; id=&quot;legends&quot;>




      <as-legend
        heading=&quot;num_bike_dropoffs&quot;
        description=&quot;&quot;>
        <as-legend-size-continuous-point id=&quot;layer0_map0_legend&quot; slot=&quot;legends&quot;></as-legend-size-continuous-point>

      </as-legend>



      <as-legend
        heading=&quot;num_bike_pickups&quot;
        description=&quot;&quot;>
        <as-legend-size-continuous-point id=&quot;layer1_map0_legend&quot; slot=&quot;legends&quot;></as-legend-size-continuous-point>

      </as-legend>


</div>
            </div> <!-- as-panel -->
          </div> <!-- as-map-panels -->

      </div> <!-- as-map-area -->
    </main> <!-- as-main -->
  </as-responsive-content>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>

<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  document
  .querySelector('as-responsive-content')
  .addEventListener('ready', () => {
    const basecolor = '';
    const basemap = 'Positron';
    const bounds = [[-77.1560030523493, 38.8341116095337], [-77.0492176172532, 38.8989419527577]];
    const camera = null;
    const default_legend = 'False' === 'true';
    const has_legends = 'true' === 'true';
    const is_static = 'None' === 'true';
    const layers = [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;v712bf3&quot;, &quot;title&quot;: &quot;num_bike_pickups&quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;width&quot;, &quot;title&quot;: &quot;num_bike_pickups&quot;, &quot;type&quot;: &quot;size-continuous-point&quot;}, &quot;query&quot;: &quot;\nSELECT\n    _ends.num_bike_dropoffs,\n    _starts.num_bike_pickups,\n    abs(_ends.num_bike_dropoffs - _starts.num_bike_pickups) as diff,\n    CASE WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups \u003e 0 THEN 1\n         WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups = 0 THEN 0\n         ELSE -1 END as diff_sign,\n    _ends.num_bike_dropoffs - _starts.num_bike_pickups as diff_relative,\n    _starts.station_id,\n    row_number() OVER () as cartodb_id,\n    ST_X(_starts.the_geom) as longitude,\n    ST_Y(_starts.the_geom) as latitude,\n    _starts.the_geom,\n    ST_Transform(_starts.the_geom, 3857) as the_geom_webmercator\nFROM\n    (SELECT\n      count(u.*) as num_bike_dropoffs,\n      u.end_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id\n    FROM capital_bikeshare_stations_points_arlington as s\n    JOIN capitalbikeshare_tripdata_201907 as u\n    ON u.end_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4) as _ends\nJOIN\n    (SELECT\n      count(u.*) as num_bike_pickups,\n      u.start_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id\n    FROM capitalbikeshare_tripdata_201907 as u\n    JOIN capital_bikeshare_stations_points_arlington as s\n    ON u.start_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4) as _starts\nON _ends.station_id = _starts.station_id\n&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v712bf3: $num_bike_pickups\ncolor: opacity(#FFB927, 0.8)\nwidth: ramp(linear(sqrt($num_bike_pickups), sqrt(globalMin($num_bike_pickups)), sqrt(globalMax($num_bike_pickups))), [2, 40])\nstrokeWidth: ramp(linear(zoom(),0,18),[0,1])\nstrokeColor: opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))\nfilter: 1\n&quot;, &quot;widgets&quot;: []}, {&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;vf4e123&quot;, &quot;title&quot;: &quot;num_bike_dropoffs&quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;width&quot;, &quot;title&quot;: &quot;num_bike_dropoffs&quot;, &quot;type&quot;: &quot;size-continuous-point&quot;}, &quot;query&quot;: &quot;\nSELECT\n    _ends.num_bike_dropoffs,\n    _starts.num_bike_pickups,\n    abs(_ends.num_bike_dropoffs - _starts.num_bike_pickups) as diff,\n    CASE WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups \u003e 0 THEN 1\n         WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups = 0 THEN 0\n         ELSE -1 END as diff_sign,\n    _ends.num_bike_dropoffs - _starts.num_bike_pickups as diff_relative,\n    _starts.station_id,\n    row_number() OVER () as cartodb_id,\n    ST_X(_starts.the_geom) as longitude,\n    ST_Y(_starts.the_geom) as latitude,\n    _starts.the_geom,\n    ST_Transform(_starts.the_geom, 3857) as the_geom_webmercator\nFROM\n    (SELECT\n      count(u.*) as num_bike_dropoffs,\n      u.end_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id\n    FROM capital_bikeshare_stations_points_arlington as s\n    JOIN capitalbikeshare_tripdata_201907 as u\n    ON u.end_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4) as _ends\nJOIN\n    (SELECT\n      count(u.*) as num_bike_pickups,\n      u.start_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id\n    FROM capitalbikeshare_tripdata_201907 as u\n    JOIN capital_bikeshare_stations_points_arlington as s\n    ON u.start_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4) as _starts\nON _ends.station_id = _starts.station_id\n&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@vf4e123: $num_bike_dropoffs\ncolor: opacity(purple, 0.8)\nwidth: ramp(linear(sqrt($num_bike_dropoffs), sqrt(globalMin($num_bike_dropoffs)), sqrt(globalMax($num_bike_dropoffs))), [2, 40])\nstrokeWidth: ramp(linear(zoom(),0,18),[0,1])\nstrokeColor: opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))\nfilter: 1\n&quot;, &quot;widgets&quot;: []}];
    const mapboxtoken = '';
    const show_info = 'None' === 'true';

    init({
      basecolor,
      basemap,
      bounds,
      camera,
      defaultLegend: default_legend,
      has_legends: has_legends,
      is_static: is_static,
      layers,
      mapboxtoken,
      showInfo: show_info
    });
});
</script>
</html>
">

</iframe>




```python
pickup_dropoff_points = 'capital_bikeshare_july_2019_pickup_dropoff'
Dataset(sql).upload(table_name=pickup_dropoff_points)
```




    <cartoframes.data.dataset.Dataset at 0x11bac6090>




```python
Map([
    size_continuous_layer(pickup_dropoff_points, 'num_bike_pickups'),
#     color_continuous_layer(Dataset(sql), 'diff_relative')
    size_continuous_layer(pickup_dropoff_points, 'num_bike_dropoffs', color='purple')  
])
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 632px;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>None</title>
  <meta name=&quot;description&quot; content=&quot;None&quot;>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
  <style>
  as-widget-header .as-widget-header__header {
    margin-bottom: 8px;
  }

  as-widget-header .as-widget-header__subheader {
    margin-bottom: 12px;
  }

  as-category-widget {
    max-height: 250px;
  }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <img id=&quot;map-image&quot; class=&quot;map-image&quot; alt='Static map image' />
  <as-responsive-content id=&quot;main-container&quot;>

    <main class=&quot;as-main&quot;>
      <div class=&quot;as-map-area&quot;>
        <div id=&quot;map&quot; class=&quot;map&quot;></div>


          <div class=&quot;as-map-panels&quot; data-name=&quot;Legends&quot;>
            <div class=&quot;as-panel as-panel--left as-panel--top&quot;>


<div class=&quot;as-panel__element&quot; id=&quot;legends&quot;>




      <as-legend
        heading=&quot;num_bike_dropoffs&quot;
        description=&quot;&quot;>
        <as-legend-size-continuous-point id=&quot;layer0_map0_legend&quot; slot=&quot;legends&quot;></as-legend-size-continuous-point>

      </as-legend>



      <as-legend
        heading=&quot;num_bike_pickups&quot;
        description=&quot;&quot;>
        <as-legend-size-continuous-point id=&quot;layer1_map0_legend&quot; slot=&quot;legends&quot;></as-legend-size-continuous-point>

      </as-legend>


</div>
            </div> <!-- as-panel -->
          </div> <!-- as-map-panels -->

      </div> <!-- as-map-area -->
    </main> <!-- as-main -->
  </as-responsive-content>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>

<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  document
  .querySelector('as-responsive-content')
  .addEventListener('ready', () => {
    const basecolor = '';
    const basemap = 'Positron';
    const bounds = [[-77.1560030523493, 38.8341116095337], [-77.0492176172532, 38.8989419527577]];
    const camera = null;
    const default_legend = 'False' === 'true';
    const has_legends = 'true' === 'true';
    const is_static = 'None' === 'true';
    const layers = [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;v712bf3&quot;, &quot;title&quot;: &quot;num_bike_pickups&quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;width&quot;, &quot;title&quot;: &quot;num_bike_pickups&quot;, &quot;type&quot;: &quot;size-continuous-point&quot;}, &quot;query&quot;: &quot;SELECT * FROM \&quot;eschbacher\&quot;.\&quot;capital_bikeshare_july_2019_pickup_dropoff\&quot;&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v712bf3: $num_bike_pickups\ncolor: opacity(#FFB927, 0.8)\nwidth: ramp(linear(sqrt($num_bike_pickups), sqrt(globalMin($num_bike_pickups)), sqrt(globalMax($num_bike_pickups))), [2, 40])\nstrokeWidth: ramp(linear(zoom(),0,18),[0,1])\nstrokeColor: opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))\nfilter: 1\n&quot;, &quot;widgets&quot;: []}, {&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;vf4e123&quot;, &quot;title&quot;: &quot;num_bike_dropoffs&quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;width&quot;, &quot;title&quot;: &quot;num_bike_dropoffs&quot;, &quot;type&quot;: &quot;size-continuous-point&quot;}, &quot;query&quot;: &quot;SELECT * FROM \&quot;eschbacher\&quot;.\&quot;capital_bikeshare_july_2019_pickup_dropoff\&quot;&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@vf4e123: $num_bike_dropoffs\ncolor: opacity(purple, 0.8)\nwidth: ramp(linear(sqrt($num_bike_dropoffs), sqrt(globalMin($num_bike_dropoffs)), sqrt(globalMax($num_bike_dropoffs))), [2, 40])\nstrokeWidth: ramp(linear(zoom(),0,18),[0,1])\nstrokeColor: opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))\nfilter: 1\n&quot;, &quot;widgets&quot;: []}];
    const mapboxtoken = '';
    const show_info = 'None' === 'true';

    init({
      basecolor,
      basemap,
      bounds,
      camera,
      defaultLegend: default_legend,
      has_legends: has_legends,
      is_static: is_static,
      layers,
      mapboxtoken,
      showInfo: show_info
    });
});
</script>
</html>
">

</iframe>




```python
from cartoframes.viz import WidgetList
value = 'diff'
title = 'Drop Offs '
diff_map = Layer(
        sql,
        style={
            'point': {
                'width': 'ramp(linear(sqrt(${0}), sqrt(globalMin(${0})), sqrt(globalMax(${0}))), {1})'.format(
                    value, [2, 40]),
                'color': 'opacity(ramp(${0}, antique), 0.8)'.format(
                    'diff_sign', 'diff_sign'),
                'strokeColor': 'opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))',
            }
        },
        popup={
            'hover': [{
                'title': title,
                'value': f'$diff_sign * ${value}'
            }]
        },
        legend={
            'type': {
                'point': 'size-continuous-point',
                'line': 'size-continuous-line',
                'polygon': 'size-continuous-polygon'
            },
            'title': title,
            'description': '',
            'footer': ''
        }
    )
diff_map
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 632px;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>None</title>
  <meta name=&quot;description&quot; content=&quot;None&quot;>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
  <style>
  as-widget-header .as-widget-header__header {
    margin-bottom: 8px;
  }

  as-widget-header .as-widget-header__subheader {
    margin-bottom: 12px;
  }

  as-category-widget {
    max-height: 250px;
  }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <img id=&quot;map-image&quot; class=&quot;map-image&quot; alt='Static map image' />
  <as-responsive-content id=&quot;main-container&quot;>

    <main class=&quot;as-main&quot;>
      <div class=&quot;as-map-area&quot;>
        <div id=&quot;map&quot; class=&quot;map&quot;></div>


          <div class=&quot;as-map-panels&quot; data-name=&quot;Legends&quot;>
            <div class=&quot;as-panel as-panel--left as-panel--top&quot;>


<div class=&quot;as-panel__element&quot; id=&quot;legends&quot;>




      <as-legend
        heading=&quot;Drop Offs &quot;
        description=&quot;&quot;>
        <as-legend-size-continuous-point id=&quot;layer0_map0_legend&quot; slot=&quot;legends&quot;></as-legend-size-continuous-point>

      </as-legend>


</div>
            </div> <!-- as-panel -->
          </div> <!-- as-map-panels -->

      </div> <!-- as-map-area -->
    </main> <!-- as-main -->
  </as-responsive-content>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>

<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  document
  .querySelector('as-responsive-content')
  .addEventListener('ready', () => {
    const basecolor = '';
    const basemap = 'Positron';
    const bounds = [[-77.1560030523493, 38.8341116095337], [-77.0492176172532, 38.8989419527577]];
    const camera = null;
    const default_legend = 'False' === 'true';
    const has_legends = 'true' === 'true';
    const is_static = 'None' === 'true';
    const layers = [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;v7dd36f&quot;, &quot;title&quot;: &quot;Drop Offs &quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;width&quot;, &quot;title&quot;: &quot;Drop Offs &quot;, &quot;type&quot;: &quot;size-continuous-point&quot;}, &quot;query&quot;: &quot;\nSELECT\n    _ends.num_bike_dropoffs,\n    _starts.num_bike_pickups,\n    abs(_ends.num_bike_dropoffs - _starts.num_bike_pickups) as diff,\n    CASE WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups \u003e 0 THEN 1\n         WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups = 0 THEN 0\n         ELSE -1 END as diff_sign,\n    _ends.num_bike_dropoffs - _starts.num_bike_pickups as diff_relative,\n    _starts.station_id,\n    row_number() OVER () as cartodb_id,\n    ST_X(_starts.the_geom) as longitude,\n    ST_Y(_starts.the_geom) as latitude,\n    _starts.the_geom,\n    ST_Transform(_starts.the_geom, 3857) as the_geom_webmercator,\n    _ends.day_of_month::numeric as day_of_month\nFROM\n    (SELECT\n      count(u.*) as num_bike_dropoffs,\n      u.end_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id,\n      EXTRACT(DAY FROM end_date) as day_of_month\n    FROM capital_bikeshare_stations_points_arlington as s\n    JOIN capitalbikeshare_tripdata_201907 as u\n    ON u.end_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4, 5) as _ends\nJOIN\n    (SELECT\n      count(u.*) as num_bike_pickups,\n      u.start_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id,\n      EXTRACT(DAY FROM start_date) as day_of_month\n    FROM capitalbikeshare_tripdata_201907 as u\n    JOIN capital_bikeshare_stations_points_arlington as s\n    ON u.start_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4, 5) as _starts\nON _ends.station_id = _starts.station_id and _ends.day_of_month = _starts.day_of_month\n&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v7dd36f: $diff_sign * $diff\ncolor: opacity(ramp($diff_sign, antique), 0.8)\nwidth: ramp(linear(sqrt($diff), sqrt(globalMin($diff)), sqrt(globalMax($diff))), [2, 40])\nstrokeWidth: ramp(linear(zoom(),0,18),[0,1])\nstrokeColor: opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))\n&quot;, &quot;widgets&quot;: []}];
    const mapboxtoken = '';
    const show_info = 'None' === 'true';

    init({
      basecolor,
      basemap,
      bounds,
      camera,
      defaultLegend: default_legend,
      has_legends: has_legends,
      is_static: is_static,
      layers,
      mapboxtoken,
      showInfo: show_info
    });
});
</script>
</html>
">

</iframe>



### Animating pickups/dropoffs over time


```python
sql = f'''
SELECT
    _ends.num_bike_dropoffs,
    _starts.num_bike_pickups,
    abs(_ends.num_bike_dropoffs - _starts.num_bike_pickups) as diff,
    CASE WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups > 0 THEN 1
         WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups = 0 THEN 0
         ELSE -1 END as diff_sign,
    _ends.num_bike_dropoffs - _starts.num_bike_pickups as diff_relative,
    _starts.station_id,
    row_number() OVER () as cartodb_id,
    ST_X(_starts.the_geom) as longitude,
    ST_Y(_starts.the_geom) as latitude,
    _starts.the_geom,
    ST_Transform(_starts.the_geom, 3857) as the_geom_webmercator,
    _ends.day_of_month::numeric as day_of_month
FROM
    (SELECT
      count(u.*) as num_bike_dropoffs,
      u.end_station_number::int as station_id,
      s.the_geom,
      s.cartodb_id,
      EXTRACT(DAY FROM end_date) as day_of_month
    FROM {station_points} as s
    JOIN {usage_data} as u
    ON u.end_station_number::int = s.gisid::int
    GROUP BY 2, 3, 4, 5) as _ends
JOIN
    (SELECT
      count(u.*) as num_bike_pickups,
      u.start_station_number::int as station_id,
      s.the_geom,
      s.cartodb_id,
      EXTRACT(DAY FROM start_date) as day_of_month
    FROM {usage_data} as u
    JOIN {station_points} as s
    ON u.start_station_number::int = s.gisid::int
    GROUP BY 2, 3, 4, 5) as _starts
ON _ends.station_id = _starts.station_id and _ends.day_of_month = _starts.day_of_month
'''

size_continuous_layer(Dataset(sql), 'diff', animate='day_of_month')
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 632px;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>None</title>
  <meta name=&quot;description&quot; content=&quot;None&quot;>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
  <style>
  as-widget-header .as-widget-header__header {
    margin-bottom: 8px;
  }

  as-widget-header .as-widget-header__subheader {
    margin-bottom: 12px;
  }

  as-category-widget {
    max-height: 250px;
  }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <img id=&quot;map-image&quot; class=&quot;map-image&quot; alt='Static map image' />
  <as-responsive-content id=&quot;main-container&quot;>



<aside class=&quot;as-sidebar as-sidebar--right&quot; id=&quot;widgets&quot; data-name=&quot;Widgets&quot;>




          <div class=&quot;as-box&quot;>
            <section class=&quot;as-body&quot;>

      <as-time-series-widget
  animated
  id=&quot;layer0_widget0&quot;
  time-format=&quot;auto&quot;
  description=&quot;&quot;
  heading=&quot;Animation&quot;>
</as-time-series-widget>

  </section>
          </div>



</aside>

    <main class=&quot;as-main&quot;>
      <div class=&quot;as-map-area&quot;>
        <div id=&quot;map&quot; class=&quot;map&quot;></div>


          <div class=&quot;as-map-panels&quot; data-name=&quot;Legends&quot;>
            <div class=&quot;as-panel as-panel--left as-panel--top&quot;>


<div class=&quot;as-panel__element&quot; id=&quot;legends&quot;>




      <as-legend
        heading=&quot;diff&quot;
        description=&quot;&quot;>
        <as-legend-size-continuous-point id=&quot;layer0_map0_legend&quot; slot=&quot;legends&quot;></as-legend-size-continuous-point>

      </as-legend>


</div>
            </div> <!-- as-panel -->
          </div> <!-- as-map-panels -->

      </div> <!-- as-map-area -->
    </main> <!-- as-main -->
  </as-responsive-content>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>

<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  document
  .querySelector('as-responsive-content')
  .addEventListener('ready', () => {
    const basecolor = '';
    const basemap = 'Positron';
    const bounds = [[-77.1560030523493, 38.8341116095337], [-77.0492176172532, 38.8989419527577]];
    const camera = null;
    const default_legend = 'False' === 'true';
    const has_legends = 'true' === 'true';
    const is_static = 'None' === 'true';
    const layers = [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;width&quot;, &quot;title&quot;: &quot;diff&quot;, &quot;type&quot;: &quot;size-continuous-point&quot;}, &quot;query&quot;: &quot;\nSELECT\n    _ends.num_bike_dropoffs,\n    _starts.num_bike_pickups,\n    abs(_ends.num_bike_dropoffs - _starts.num_bike_pickups) as diff,\n    CASE WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups \u003e 0 THEN 1\n         WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups = 0 THEN 0\n         ELSE -1 END as diff_sign,\n    _ends.num_bike_dropoffs - _starts.num_bike_pickups as diff_relative,\n    _starts.station_id,\n    row_number() OVER () as cartodb_id,\n    ST_X(_starts.the_geom) as longitude,\n    ST_Y(_starts.the_geom) as latitude,\n    _starts.the_geom,\n    ST_Transform(_starts.the_geom, 3857) as the_geom_webmercator,\n    _ends.day_of_month::numeric as day_of_month\nFROM\n    (SELECT\n      count(u.*) as num_bike_dropoffs,\n      u.end_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id,\n      EXTRACT(DAY FROM end_date) as day_of_month\n    FROM capital_bikeshare_stations_points_arlington as s\n    JOIN capitalbikeshare_tripdata_201907 as u\n    ON u.end_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4, 5) as _ends\nJOIN\n    (SELECT\n      count(u.*) as num_bike_pickups,\n      u.start_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id,\n      EXTRACT(DAY FROM start_date) as day_of_month\n    FROM capitalbikeshare_tripdata_201907 as u\n    JOIN capital_bikeshare_stations_points_arlington as s\n    ON u.start_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4, 5) as _starts\nON _ends.station_id = _starts.station_id and _ends.day_of_month = _starts.day_of_month\n&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v80906c: $day_of_month\ncolor: opacity(#FFB927, 0.8)\nwidth: ramp(linear(sqrt($diff), sqrt(globalMin($diff)), sqrt(globalMax($diff))), [2, 40])\nstrokeWidth: ramp(linear(zoom(),0,18),[0,1])\nstrokeColor: opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))\nfilter: animation(linear($day_of_month), 20, fade(1,1))\n&quot;, &quot;widgets&quot;: [{&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;has_bridge&quot;: true, &quot;options&quot;: {&quot;readOnly&quot;: false}, &quot;prop&quot;: &quot;filter&quot;, &quot;title&quot;: &quot;Animation&quot;, &quot;type&quot;: &quot;time-series&quot;, &quot;value&quot;: &quot;day_of_month&quot;, &quot;variable_name&quot;: &quot;v80906c&quot;}]}];
    const mapboxtoken = '';
    const show_info = 'None' === 'true';

    init({
      basecolor,
      basemap,
      bounds,
      camera,
      defaultLegend: default_legend,
      has_legends: has_legends,
      is_static: is_static,
      layers,
      mapboxtoken,
      showInfo: show_info
    });
});
</script>
</html>
">

</iframe>




```python
cols = set(Dataset(pickup_dropoff_points).download(limit=5).columns) - {'the_geom', 'cartodb_id', 'the_geom_webmercator'}
sql = f'''
CREATE TABLE {pickup_dropoff_points}_isochrone AS
SELECT ST_MakeValid((cdb_isochrone(the_geom, 'walk', Array[600])).the_geom) as the_geom, cartodb_id,
    {','.join(cols)}
FROM {pickup_dropoff_points};
SELECT CDB_Cartodbfytable('eschbacher', '{pickup_dropoff_points}_isochrone');
'''

sql_client.execute(sql)
```

```python
Layer(f'{pickup_dropoff_points}_isochrone')
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 632px;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>None</title>
  <meta name=&quot;description&quot; content=&quot;None&quot;>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
  <style>
  as-widget-header .as-widget-header__header {
    margin-bottom: 8px;
  }

  as-widget-header .as-widget-header__subheader {
    margin-bottom: 12px;
  }

  as-category-widget {
    max-height: 250px;
  }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <img id=&quot;map-image&quot; class=&quot;map-image&quot; alt='Static map image' />
  <as-responsive-content id=&quot;main-container&quot;>

    <main class=&quot;as-main&quot;>
      <div class=&quot;as-map-area&quot;>
        <div id=&quot;map&quot; class=&quot;map&quot;></div>


      </div> <!-- as-map-area -->
    </main> <!-- as-main -->
  </as-responsive-content>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>

<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  document
  .querySelector('as-responsive-content')
  .addEventListener('ready', () => {
    const basecolor = '';
    const basemap = 'Positron';
    const bounds = [[-77.16358, 38.82744], [-77.04366, 38.90519]];
    const camera = null;
    const default_legend = 'False' === 'true';
    const has_legends = 'False' === 'true';
    const is_static = 'None' === 'true';
    const layers = [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [], &quot;legend&quot;: {}, &quot;query&quot;: &quot;SELECT * FROM \&quot;eschbacher\&quot;.\&quot;capital_bikeshare_july_2019_pickup_dropoff_isochrone\&quot;&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;color: hex(\&quot;#826DBA\&quot;)\nstrokeWidth: ramp(linear(zoom(),2,18),[0.5,1])\nstrokeColor: opacity(#2c2c2c,ramp(linear(zoom(),2,18),[0.2,0.6]))\n&quot;, &quot;widgets&quot;: []}];
    const mapboxtoken = '';
    const show_info = 'None' === 'true';

    init({
      basecolor,
      basemap,
      bounds,
      camera,
      defaultLegend: default_legend,
      has_legends: has_legends,
      is_static: is_static,
      layers,
      mapboxtoken,
      showInfo: show_info
    });
});
</script>
</html>
">

</iframe>




```python
from cartoframes.client import DataObsClient
do = DataObsClient(creds)
```

```python
do_aug = do.augment(
    f'{pickup_dropoff_points}_isochrone',
    [{"numer_id": "us.census.acs.B23006001",
         "denom_id": "us.census.acs.B01003001",
         "normalization": "denominated"}]
)
```

```python
do_aug.dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diff</th>
      <th>num_bike_pickups</th>
      <th>num_bike_dropoffs</th>
      <th>station_id</th>
      <th>the_geom</th>
      <th>diff_relative</th>
      <th>latitude</th>
      <th>diff_sign</th>
      <th>longitude</th>
      <th>pop_25_64_2006_2010_by_total_pop</th>
      <th>pop_25_64_2010_2014_by_total_pop</th>
      <th>pop_25_64_2011_2015_by_total_pop</th>
    </tr>
    <tr>
      <th>cartodb_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>26.0</td>
      <td>204.0</td>
      <td>178.0</td>
      <td>31000.0</td>
      <td>0106000020E61000000100000001030000000100000033...</td>
      <td>-26.0</td>
      <td>38.858726</td>
      <td>-1.0</td>
      <td>-77.053144</td>
      <td>0.769188</td>
      <td>0.756989</td>
      <td>0.747417</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54.0</td>
      <td>276.0</td>
      <td>222.0</td>
      <td>31001.0</td>
      <td>0106000020E61000000100000001030000000100000033...</td>
      <td>-54.0</td>
      <td>38.857216</td>
      <td>-1.0</td>
      <td>-77.053738</td>
      <td>0.758869</td>
      <td>0.746634</td>
      <td>0.738867</td>
    </tr>
    <tr>
      <th>3</th>
      <td>129.0</td>
      <td>710.0</td>
      <td>839.0</td>
      <td>31002.0</td>
      <td>0106000020E61000000400000001030000000100000006...</td>
      <td>129.0</td>
      <td>38.856372</td>
      <td>1.0</td>
      <td>-77.049218</td>
      <td>0.758723</td>
      <td>0.746481</td>
      <td>0.738368</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>489.0</td>
      <td>487.0</td>
      <td>31003.0</td>
      <td>0106000020E61000000200000001030000000100000027...</td>
      <td>-2.0</td>
      <td>38.860167</td>
      <td>-1.0</td>
      <td>-77.049614</td>
      <td>0.776694</td>
      <td>0.764732</td>
      <td>0.757629</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>227.0</td>
      <td>230.0</td>
      <td>31004.0</td>
      <td>0106000020E61000000100000001030000000100000033...</td>
      <td>3.0</td>
      <td>38.857937</td>
      <td>1.0</td>
      <td>-77.059552</td>
      <td>0.738381</td>
      <td>0.725455</td>
      <td>0.722892</td>
    </tr>
  </tbody>
</table>
</div>




```python
do_aug.upload(table_name=f'{pickup_dropoff_points}_isochrone_augmented')
```




    <cartoframes.data.dataset.Dataset at 0x11b0479d0>




```python
color_continuous_layer(f'{pickup_dropoff_points}_isochrone_augmented', 'pop_25_64_2011_2015_by_total_pop')
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 632px;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>None</title>
  <meta name=&quot;description&quot; content=&quot;None&quot;>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
  <style>
  as-widget-header .as-widget-header__header {
    margin-bottom: 8px;
  }

  as-widget-header .as-widget-header__subheader {
    margin-bottom: 12px;
  }

  as-category-widget {
    max-height: 250px;
  }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <img id=&quot;map-image&quot; class=&quot;map-image&quot; alt='Static map image' />
  <as-responsive-content id=&quot;main-container&quot;>

    <main class=&quot;as-main&quot;>
      <div class=&quot;as-map-area&quot;>
        <div id=&quot;map&quot; class=&quot;map&quot;></div>


          <div class=&quot;as-map-panels&quot; data-name=&quot;Legends&quot;>
            <div class=&quot;as-panel as-panel--left as-panel--top&quot;>


<div class=&quot;as-panel__element&quot; id=&quot;legends&quot;>




      <as-legend
        heading=&quot;pop_25_64_2011_2015_by_total_pop&quot;
        description=&quot;&quot;>
        <as-legend-color-continuous-polygon id=&quot;layer0_map0_legend&quot; slot=&quot;legends&quot;></as-legend-color-continuous-polygon>

      </as-legend>


</div>
            </div> <!-- as-panel -->
          </div> <!-- as-map-panels -->

      </div> <!-- as-map-area -->
    </main> <!-- as-main -->
  </as-responsive-content>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>

<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  document
  .querySelector('as-responsive-content')
  .addEventListener('ready', () => {
    const basecolor = '';
    const basemap = 'Positron';
    const bounds = [[-77.16358, 38.82744], [-77.04366, 38.90519]];
    const camera = null;
    const default_legend = 'False' === 'true';
    const has_legends = 'true' === 'true';
    const is_static = 'None' === 'true';
    const layers = [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;v6e178d&quot;, &quot;title&quot;: &quot;pop_25_64_2011_2015_by_total_pop&quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;color&quot;, &quot;title&quot;: &quot;pop_25_64_2011_2015_by_total_pop&quot;, &quot;type&quot;: &quot;color-continuous-polygon&quot;}, &quot;query&quot;: &quot;SELECT * FROM \&quot;eschbacher\&quot;.\&quot;capital_bikeshare_july_2019_pickup_dropoff_isochrone_augmented\&quot;&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v6e178d: $pop_25_64_2011_2015_by_total_pop\ncolor: opacity(ramp(linear($pop_25_64_2011_2015_by_total_pop), bluyl), 0.9)\nstrokeWidth: ramp(linear(zoom(),2,18),[0.5,1])\nstrokeColor: opacity(#2c2c2c,ramp(linear(zoom(),2,18),[0.2,0.6]))\nfilter: 1\n&quot;, &quot;widgets&quot;: []}];
    const mapboxtoken = '';
    const show_info = 'None' === 'true';

    init({
      basecolor,
      basemap,
      bounds,
      camera,
      defaultLegend: default_legend,
      has_legends: has_legends,
      is_static: is_static,
      layers,
      mapboxtoken,
      showInfo: show_info
    });
});
</script>
</html>
">

</iframe>



### Run a Model


```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipeline.fit(do_aug.dataframe[['num_bike_pickups', 'num_bike_dropoffs', 'pop_25_64_2011_2015_by_total_pop']])
```




    Pipeline(memory=None,
             steps=[('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('kmeans',
                     KMeans(algorithm='auto', copy_x=True, init='k-means++',
                            max_iter=300, n_clusters=5, n_init=10, n_jobs=None,
                            precompute_distances='auto', random_state=None,
                            tol=0.0001, verbose=0))],
             verbose=False)



### Add results to dataset


```python
do_aug.dataframe['labels'] = pipeline.steps[1][1].labels_.astype(str)
```

```python
do_aug.dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diff</th>
      <th>num_bike_pickups</th>
      <th>num_bike_dropoffs</th>
      <th>station_id</th>
      <th>the_geom</th>
      <th>diff_relative</th>
      <th>latitude</th>
      <th>diff_sign</th>
      <th>longitude</th>
      <th>pop_25_64_2006_2010_by_total_pop</th>
      <th>pop_25_64_2010_2014_by_total_pop</th>
      <th>pop_25_64_2011_2015_by_total_pop</th>
      <th>labels</th>
    </tr>
    <tr>
      <th>cartodb_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>26.0</td>
      <td>204.0</td>
      <td>178.0</td>
      <td>31000.0</td>
      <td>0106000020E61000000100000001030000000100000033...</td>
      <td>-26.0</td>
      <td>38.858726</td>
      <td>-1.0</td>
      <td>-77.053144</td>
      <td>0.769188</td>
      <td>0.756989</td>
      <td>0.747417</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54.0</td>
      <td>276.0</td>
      <td>222.0</td>
      <td>31001.0</td>
      <td>0106000020E61000000100000001030000000100000033...</td>
      <td>-54.0</td>
      <td>38.857216</td>
      <td>-1.0</td>
      <td>-77.053738</td>
      <td>0.758869</td>
      <td>0.746634</td>
      <td>0.738867</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>129.0</td>
      <td>710.0</td>
      <td>839.0</td>
      <td>31002.0</td>
      <td>0106000020E61000000400000001030000000100000006...</td>
      <td>129.0</td>
      <td>38.856372</td>
      <td>1.0</td>
      <td>-77.049218</td>
      <td>0.758723</td>
      <td>0.746481</td>
      <td>0.738368</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>489.0</td>
      <td>487.0</td>
      <td>31003.0</td>
      <td>0106000020E61000000200000001030000000100000027...</td>
      <td>-2.0</td>
      <td>38.860167</td>
      <td>-1.0</td>
      <td>-77.049614</td>
      <td>0.776694</td>
      <td>0.764732</td>
      <td>0.757629</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>227.0</td>
      <td>230.0</td>
      <td>31004.0</td>
      <td>0106000020E61000000100000001030000000100000033...</td>
      <td>3.0</td>
      <td>38.857937</td>
      <td>1.0</td>
      <td>-77.059552</td>
      <td>0.738381</td>
      <td>0.725455</td>
      <td>0.722892</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
do_aug.upload(table_name='demo_augmentation', if_exists='replace')
```




    <cartoframes.data.dataset.Dataset at 0x11b0479d0>




```python
results_map = Map(
    color_category_layer('demo_augmentation', 'labels', widget=True, palette='prism'),
    
)
results_map
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 632px;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>None</title>
  <meta name=&quot;description&quot; content=&quot;None&quot;>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
  <style>
  as-widget-header .as-widget-header__header {
    margin-bottom: 8px;
  }

  as-widget-header .as-widget-header__subheader {
    margin-bottom: 12px;
  }

  as-category-widget {
    max-height: 250px;
  }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <img id=&quot;map-image&quot; class=&quot;map-image&quot; alt='Static map image' />
  <as-responsive-content id=&quot;main-container&quot;>



<aside class=&quot;as-sidebar as-sidebar--right&quot; id=&quot;widgets&quot; data-name=&quot;Widgets&quot;>




          <div class=&quot;as-box&quot;>
            <section class=&quot;as-body&quot;>

      <as-category-widget
  id=&quot;layer0_widget0&quot;
  description=&quot;&quot;
  heading=&quot;Categories&quot;>
</as-category-widget>

  </section>
          </div>



</aside>

    <main class=&quot;as-main&quot;>
      <div class=&quot;as-map-area&quot;>
        <div id=&quot;map&quot; class=&quot;map&quot;></div>


          <div class=&quot;as-map-panels&quot; data-name=&quot;Legends&quot;>
            <div class=&quot;as-panel as-panel--left as-panel--top&quot;>


<div class=&quot;as-panel__element&quot; id=&quot;legends&quot;>




      <as-legend
        heading=&quot;labels&quot;
        description=&quot;&quot;>
        <as-legend-color-category-polygon id=&quot;layer0_map0_legend&quot; slot=&quot;legends&quot;></as-legend-color-category-polygon>

      </as-legend>


</div>
            </div> <!-- as-panel -->
          </div> <!-- as-map-panels -->

      </div> <!-- as-map-area -->
    </main> <!-- as-main -->
  </as-responsive-content>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>

<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  document
  .querySelector('as-responsive-content')
  .addEventListener('ready', () => {
    const basecolor = '';
    const basemap = 'Positron';
    const bounds = [[-77.16358, 38.82744], [-77.04366, 38.90519]];
    const camera = null;
    const default_legend = 'False' === 'true';
    const has_legends = 'true' === 'true';
    const is_static = 'None' === 'true';
    const layers = [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;v8718bf&quot;, &quot;title&quot;: &quot;labels&quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;color&quot;, &quot;title&quot;: &quot;labels&quot;, &quot;type&quot;: &quot;color-category-polygon&quot;}, &quot;query&quot;: &quot;SELECT * FROM \&quot;eschbacher\&quot;.\&quot;demo_augmentation\&quot;&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v8718bf: $labels\n@vc7775e: $labels\ncolor: opacity(ramp(top($labels, 11), prism), 0.9)\nstrokeWidth: ramp(linear(zoom(),2,18),[0.5,1])\nstrokeColor: opacity(#2c2c2c,ramp(linear(zoom(),2,18),[0.2,0.6]))\nfilter: 1\n&quot;, &quot;widgets&quot;: [{&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;has_bridge&quot;: true, &quot;options&quot;: {&quot;readOnly&quot;: false}, &quot;prop&quot;: &quot;&quot;, &quot;title&quot;: &quot;Categories&quot;, &quot;type&quot;: &quot;category&quot;, &quot;value&quot;: &quot;labels&quot;, &quot;variable_name&quot;: &quot;vc7775e&quot;}]}];
    const mapboxtoken = '';
    const show_info = 'None' === 'true';

    init({
      basecolor,
      basemap,
      bounds,
      camera,
      defaultLegend: default_legend,
      has_legends: has_legends,
      is_static: is_static,
      layers,
      mapboxtoken,
      showInfo: show_info
    });
});
</script>
</html>
">

</iframe>




```python
results_map.publish('Station Encoding')
```




    {'id': '4f720d6c-561c-4db6-b07e-9702c8890d8b',
     'url': 'https://team.carto.com/u/eschbacher/kuviz/4f720d6c-561c-4db6-b07e-9702c8890d8b',
     'name': 'Station Encoding',
     'privacy': 'public'}



### Map Grid


```python
from cartoframes.viz import MapGrid
results_map = Map(
    color_category_layer('demo_augmentation', 'labels')
)
MapGrid([
    results_map,
    Map(diff_map)
], 2, 1, viewport={'zoom': 11})
```




<iframe
  frameborder="0"
  style="
    border: 1px solid #cfcfcf;
    width: 100%;
    height: 250px;;
    "
  srcDoc="
  <!DOCTYPE html>
<html lang=&quot;en&quot;>
<head>
  <title>CARTOframes</title>
  <meta name=&quot;viewport&quot; content=&quot;width=device-width, initial-scale=1.0&quot;>
  <meta charset=&quot;UTF-8&quot;>
  <!-- Include CARTO VL JS -->
  <script src=&quot;https://libs.cartocdn.com/carto-vl/v1.4/carto-vl.min.js&quot;></script>
  <!-- Include Mapbox GL JS -->
  <script src=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.js&quot;></script>
  <!-- Include Mapbox GL CSS -->
  <link href=&quot;https://api.tiles.mapbox.com/mapbox-gl-js/v1.0.0/mapbox-gl.css&quot; rel=&quot;stylesheet&quot; />

  <!-- Include Airship -->
  <script nomodule=&quot;&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship.js&quot;></script>
  <script type=&quot;module&quot; src=&quot;https://libs.cartocdn.com/airship-components/v2.2.0-rc.2/airship/airship.esm.js&quot;></script>
  <script src=&quot;https://libs.cartocdn.com/airship-bridge/v2.2.0-rc.2/asbridge.min.js&quot;></script>
  <link href=&quot;https://libs.cartocdn.com/airship-style/v2.2.0-rc.2/airship.min.css&quot; rel=&quot;stylesheet&quot;>
  <link href=&quot;https://libs.cartocdn.com/airship-icons/v2.2.0-rc.2/icons.css&quot; rel=&quot;stylesheet&quot;>

  <link href=&quot;https://fonts.googleapis.com/css?family=Roboto&quot; rel=&quot;stylesheet&quot; type=&quot;text/css&quot;>


  <style>
  body {
    margin: 0;
    padding: 0;
  }

  aside.as-sidebar {
    min-width: 300px;
  }

  .map-image {
    display: none;
    max-width: 100%;
    height: auto;
  }
</style>
  <style>
  .map {
    position: absolute;
    height: 100%;
    width: 100%;
  }

  .map-info {
    position: absolute;
    bottom: 0;
    padding: 0 5px;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0;
    color: rgba(0, 0, 0, 0.75);
    font-size: 12px;
    width: auto;
    height: 18px;
    font-family: 'Open Sans';
  }

  .map-footer {
    background: #F2F6F9;
    font-family: Roboto;
    font-size: 12px;
    line-height: 24px;
    color: #162945;
    text-align: center;
    z-index: 2;
  }

  .map-footer a {
    text-decoration: none;
  }

  .map-footer a:hover {
    text-decoration: underline;
  }
</style>
  <style>
.grid-map-cell p {
  font-family: Roboto;
  font-size: 14px;
}

.grid-map.as-main {
  width: 100%;
}

.grid-map-cell {
  align-items: center;
  display: flex;
  flex-direction: column;
  height: 250px;
  justify-content: center;
  width: 100%;
  padding: 8px;
  position: relative;
}

.grid-map-cell-legend {
  flex-direction: row;
}

h2.as-widget-header__header {
  font-size: 12px;
  line-height: 12px;
  padding: 8px 0;
}

.grid-map {
  background-color: silver;
  height: 75%;
  width: 85%;
}

.grid-map > .as-map-area,
.map-image > .as-map-area {
  height: 100%;
}

.grid-map-container {
  align-items: center;
  display: flex;
  flex-direction: row;
  justify-content: space-around;
  width: 100%;
  padding: 0 8px;
}

.grid-map-row {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  width: 100%;
}

.grid-legends {
  background-color: white;
}
</style>
    <style>
    #error-container {
      position: absolute;
      width: 100%;
      height: 100%;
      background-color: white;
      visibility: hidden;
      padding: 1em;
      font-family: &quot;Courier New&quot;, Courier, monospace;
      margin: 0 auto;
      font-size: 14px;
      overflow: auto;
      z-index: 1000;
      color: black;
    }

    .error-section {
      padding: 1em;
      border-radius: 5px;
      background-color: #fee;
    }

    #error-container #error-highlight {
      font-weight: bold;
      color: inherit;
    }

    #error-container #error-type {
      color: #008000;
    }

    #error-container #error-name {
      color: #ba2121;
    }

    #error-container #error-content {
      margin-top: 0.4em;
    }

    .error-details {
      margin-top: 1em;
    }

    #error-stacktrace {
      list-style: none;
    }
</style>
  <style>
    .popup-content {
      display: flex;
      flex-direction: column;
      padding: 8px;
    }

    .popup-name {
      font-size: 12px;
      font-weight: 400;
      line-height: 20px;
      margin-bottom: 4px;
    }

    .popup-value {
      font-size: 16px;
      font-weight: 600;
      line-height: 20px;
    }

    .popup-value:not(:last-of-type) {
      margin-bottom: 16px;
    }
</style>
</head>

<body class=&quot;as-app-body as-app&quot;>
  <main id=&quot;main-container&quot; class=&quot;grid-map-container&quot;>

    <div class=&quot;grid-map-row&quot;>





      <div class=&quot;grid-map-cell grid-map-cell-legend&quot;>





<div id=&quot;legends&quot; class=&quot;legends grid-legends&quot;>




      <as-legend
        class=&quot;grid-legend&quot;
        heading=&quot;labels&quot;
        description=&quot;&quot;
      >
        <as-legend-color-category-polygon id=&quot;layer0_map0_legend&quot; slot=&quot;legends&quot;></as-legend-color-category-polygon>

      </as-legend>


</div>

          <img id=&quot;map-image-0&quot; class=&quot;map-image&quot; alt=&quot;Static Map 0&quot; />
          <div class=&quot;as-main grid-map&quot; id=&quot;main-container-0&quot;>
            <div class=&quot;as-map-area&quot;>
              <div id=&quot;map-0&quot; class=&quot;map&quot;></div>
            </div> <!-- as-map-area -->
          </div> <!-- as-main -->
      </div>

    </div>

    <div class=&quot;grid-map-row&quot;>





      <div class=&quot;grid-map-cell grid-map-cell-legend&quot;>





<div id=&quot;legends&quot; class=&quot;legends grid-legends&quot;>




      <as-legend
        class=&quot;grid-legend&quot;
        heading=&quot;Drop Offs &quot;
        description=&quot;&quot;
      >
        <as-legend-size-continuous-point id=&quot;layer0_map1_legend&quot; slot=&quot;legends&quot;></as-legend-size-continuous-point>

      </as-legend>


</div>

          <img id=&quot;map-image-1&quot; class=&quot;map-image&quot; alt=&quot;Static Map 1&quot; />
          <div class=&quot;as-main grid-map&quot; id=&quot;main-container-1&quot;>
            <div class=&quot;as-map-area&quot;>
              <div id=&quot;map-1&quot; class=&quot;map&quot;></div>
            </div> <!-- as-map-area -->
          </div> <!-- as-main -->
      </div>

    </div>

  </main>



  <div id=&quot;error-container&quot; class=&quot;error&quot;>
  <p>There is a <span class=&quot;errors&quot; id=&quot;error-highlight&quot;></span>
  from the <a href=&quot;https://carto.com/developers/carto-vl/&quot; target=&quot;_blank&quot;>CARTO VL</a> library:</p>
  <section class=&quot;error-section&quot;>
    <span class=&quot;errors&quot; id=&quot;error-name&quot;></span>:
    <section id=&quot;error-content&quot;>
      <span class=&quot;errors&quot; id=&quot;error-type&quot;></span>
      <span class=&quot;errors&quot; id=&quot;error-message&quot;></span>
    </section>
  </section>

  <details class=&quot;error-details&quot;>
    <summary>StackTrace</summary>
    <ul id=&quot;error-stacktrace&quot;></ul>
  </details>
</div>
</body>
<script>
  /*
 *  base64.js
 *
 *  Licensed under the BSD 3-Clause License.
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *  References:
 *    http://en.wikipedia.org/wiki/Base64
 */
;(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined'
        ? module.exports = factory(global)
        : typeof define === 'function' && define.amd
        ? define(factory) : factory(global)
}((
    typeof self !== 'undefined' ? self
        : typeof window !== 'undefined' ? window
        : typeof global !== 'undefined' ? global
: this
), function(global) {
    'use strict';
    // existing version for noConflict()
    global = global || {};
    var _Base64 = global.Base64;
    var version = &quot;2.5.1&quot;;
    // if node.js and NOT React Native, we use Buffer
    var buffer;
    if (typeof module !== 'undefined' && module.exports) {
        try {
            buffer = eval(&quot;require('buffer').Buffer&quot;);
        } catch (err) {
            buffer = undefined;
        }
    }
    // constants
    var b64chars
        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    var b64tab = function(bin) {
        var t = {};
        for (var i = 0, l = bin.length; i < l; i++) t[bin.charAt(i)] = i;
        return t;
    }(b64chars);
    var fromCharCode = String.fromCharCode;
    // encoder stuff
    var cb_utob = function(c) {
        if (c.length < 2) {
            var cc = c.charCodeAt(0);
            return cc < 0x80 ? c
                : cc < 0x800 ? (fromCharCode(0xc0 | (cc >>> 6))
                                + fromCharCode(0x80 | (cc & 0x3f)))
                : (fromCharCode(0xe0 | ((cc >>> 12) & 0x0f))
                   + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                   + fromCharCode(0x80 | ( cc         & 0x3f)));
        } else {
            var cc = 0x10000
                + (c.charCodeAt(0) - 0xD800) * 0x400
                + (c.charCodeAt(1) - 0xDC00);
            return (fromCharCode(0xf0 | ((cc >>> 18) & 0x07))
                    + fromCharCode(0x80 | ((cc >>> 12) & 0x3f))
                    + fromCharCode(0x80 | ((cc >>>  6) & 0x3f))
                    + fromCharCode(0x80 | ( cc         & 0x3f)));
        }
    };
    var re_utob = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g;
    var utob = function(u) {
        return u.replace(re_utob, cb_utob);
    };
    var cb_encode = function(ccc) {
        var padlen = [0, 2, 1][ccc.length % 3],
        ord = ccc.charCodeAt(0) << 16
            | ((ccc.length > 1 ? ccc.charCodeAt(1) : 0) << 8)
            | ((ccc.length > 2 ? ccc.charCodeAt(2) : 0)),
        chars = [
            b64chars.charAt( ord >>> 18),
            b64chars.charAt((ord >>> 12) & 63),
            padlen >= 2 ? '=' : b64chars.charAt((ord >>> 6) & 63),
            padlen >= 1 ? '=' : b64chars.charAt(ord & 63)
        ];
        return chars.join('');
    };
    var btoa = global.btoa ? function(b) {
        return global.btoa(b);
    } : function(b) {
        return b.replace(/[\s\S]{1,3}/g, cb_encode);
    };
    var _encode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function (u) {
            return (u.constructor === buffer.constructor ? u : buffer.from(u))
                .toString('base64')
        }
        :  function (u) {
            return (u.constructor === buffer.constructor ? u : new  buffer(u))
                .toString('base64')
        }
        : function (u) { return btoa(utob(u)) }
    ;
    var encode = function(u, urisafe) {
        return !urisafe
            ? _encode(String(u))
            : _encode(String(u)).replace(/[+\/]/g, function(m0) {
                return m0 == '+' ? '-' : '_';
            }).replace(/=/g, '');
    };
    var encodeURI = function(u) { return encode(u, true) };
    // decoder stuff
    var re_btou = new RegExp([
        '[\xC0-\xDF][\x80-\xBF]',
        '[\xE0-\xEF][\x80-\xBF]{2}',
        '[\xF0-\xF7][\x80-\xBF]{3}'
    ].join('|'), 'g');
    var cb_btou = function(cccc) {
        switch(cccc.length) {
        case 4:
            var cp = ((0x07 & cccc.charCodeAt(0)) << 18)
                |    ((0x3f & cccc.charCodeAt(1)) << 12)
                |    ((0x3f & cccc.charCodeAt(2)) <<  6)
                |     (0x3f & cccc.charCodeAt(3)),
            offset = cp - 0x10000;
            return (fromCharCode((offset  >>> 10) + 0xD800)
                    + fromCharCode((offset & 0x3FF) + 0xDC00));
        case 3:
            return fromCharCode(
                ((0x0f & cccc.charCodeAt(0)) << 12)
                    | ((0x3f & cccc.charCodeAt(1)) << 6)
                    |  (0x3f & cccc.charCodeAt(2))
            );
        default:
            return  fromCharCode(
                ((0x1f & cccc.charCodeAt(0)) << 6)
                    |  (0x3f & cccc.charCodeAt(1))
            );
        }
    };
    var btou = function(b) {
        return b.replace(re_btou, cb_btou);
    };
    var cb_decode = function(cccc) {
        var len = cccc.length,
        padlen = len % 4,
        n = (len > 0 ? b64tab[cccc.charAt(0)] << 18 : 0)
            | (len > 1 ? b64tab[cccc.charAt(1)] << 12 : 0)
            | (len > 2 ? b64tab[cccc.charAt(2)] <<  6 : 0)
            | (len > 3 ? b64tab[cccc.charAt(3)]       : 0),
        chars = [
            fromCharCode( n >>> 16),
            fromCharCode((n >>>  8) & 0xff),
            fromCharCode( n         & 0xff)
        ];
        chars.length -= [0, 0, 2, 1][padlen];
        return chars.join('');
    };
    var _atob = global.atob ? function(a) {
        return global.atob(a);
    } : function(a){
        return a.replace(/\S{1,4}/g, cb_decode);
    };
    var atob = function(a) {
        return _atob(String(a).replace(/[^A-Za-z0-9\+\/]/g, ''));
    };
    var _decode = buffer ?
        buffer.from && Uint8Array && buffer.from !== Uint8Array.from
        ? function(a) {
            return (a.constructor === buffer.constructor
                    ? a : buffer.from(a, 'base64')).toString();
        }
        : function(a) {
            return (a.constructor === buffer.constructor
                    ? a : new buffer(a, 'base64')).toString();
        }
        : function(a) { return btou(_atob(a)) };
    var decode = function(a){
        return _decode(
            String(a).replace(/[-_]/g, function(m0) { return m0 == '-' ? '+' : '/' })
                .replace(/[^A-Za-z0-9\+\/]/g, '')
        );
    };
    var noConflict = function() {
        var Base64 = global.Base64;
        global.Base64 = _Base64;
        return Base64;
    };
    // export Base64
    global.Base64 = {
        VERSION: version,
        atob: atob,
        btoa: btoa,
        fromBase64: decode,
        toBase64: encode,
        utob: utob,
        encode: encode,
        encodeURI: encodeURI,
        btou: btou,
        decode: decode,
        noConflict: noConflict,
        __buffer__: buffer
    };
    // if ES5 is available, make Base64.extendString() available
    if (typeof Object.defineProperty === 'function') {
        var noEnum = function(v){
            return {value:v,enumerable:false,writable:true,configurable:true};
        };
        global.Base64.extendString = function () {
            Object.defineProperty(
                String.prototype, 'fromBase64', noEnum(function () {
                    return decode(this)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64', noEnum(function (urisafe) {
                    return encode(this, urisafe)
                }));
            Object.defineProperty(
                String.prototype, 'toBase64URI', noEnum(function () {
                    return encode(this, true)
                }));
        };
    }
    //
    // export Base64 to the namespace
    //
    if (global['Meteor']) { // Meteor.js
        Base64 = global.Base64;
    }
    // module.exports and AMD are mutually exclusive.
    // module.exports has precedence.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports.Base64 = global.Base64;
    }
    else if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], function(){ return global.Base64 });
    }
    // that's it!
    return {Base64: global.Base64}
}));
</script>
<script>
  /*!
 * html2canvas 1.0.0-rc.3 <https://html2canvas.hertzen.com>
 * Copyright (c) 2019 Niklas von Hertzen <https://hertzen.com>
 * Released under MIT License
 */
!function(A,e){&quot;object&quot;==typeof exports&&&quot;undefined&quot;!=typeof module?module.exports=e():&quot;function&quot;==typeof define&&define.amd?define(e):(A=A||self).html2canvas=e()}(this,function(){&quot;use strict&quot;;
/*! *****************************************************************************
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the Apache License, Version 2.0 (the &quot;License&quot;); you may not use
    this file except in compliance with the License. You may obtain a copy of the
    License at http://www.apache.org/licenses/LICENSE-2.0

    THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
    WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
    MERCHANTABLITY OR NON-INFRINGEMENT.

    See the Apache Version 2.0 License for specific language governing permissions
    and limitations under the License.
    ***************************************************************************** */var r=function(A,e){return(r=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(A,e){A.__proto__=e}||function(A,e){for(var t in e)e.hasOwnProperty(t)&&(A[t]=e[t])})(A,e)};function A(A,e){function t(){this.constructor=A}r(A,e),A.prototype=null===e?Object.create(e):(t.prototype=e.prototype,new t)}var K=function(){return(K=Object.assign||function(A){for(var e,t=1,r=arguments.length;t<r;t++)for(var B in e=arguments[t])Object.prototype.hasOwnProperty.call(e,B)&&(A[B]=e[B]);return A}).apply(this,arguments)};function B(n,s,o,i){return new(o||(o=Promise))(function(A,e){function t(A){try{B(i.next(A))}catch(A){e(A)}}function r(A){try{B(i.throw(A))}catch(A){e(A)}}function B(e){e.done?A(e.value):new o(function(A){A(e.value)}).then(t,r)}B((i=i.apply(n,s||[])).next())})}function b(t,r){var B,n,s,A,o={label:0,sent:function(){if(1&s[0])throw s[1];return s[1]},trys:[],ops:[]};return A={next:e(0),throw:e(1),return:e(2)},&quot;function&quot;==typeof Symbol&&(A[Symbol.iterator]=function(){return this}),A;function e(e){return function(A){return function(e){if(B)throw new TypeError(&quot;Generator is already executing.&quot;);for(;o;)try{if(B=1,n&&(s=2&e[0]?n.return:e[0]?n.throw||((s=n.return)&&s.call(n),0):n.next)&&!(s=s.call(n,e[1])).done)return s;switch(n=0,s&&(e=[2&e[0],s.value]),e[0]){case 0:case 1:s=e;break;case 4:return o.label++,{value:e[1],done:!1};case 5:o.label++,n=e[1],e=[0];continue;case 7:e=o.ops.pop(),o.trys.pop();continue;default:if(!(s=0<(s=o.trys).length&&s[s.length-1])&&(6===e[0]||2===e[0])){o=0;continue}if(3===e[0]&&(!s||e[1]>s[0]&&e[1]<s[3])){o.label=e[1];break}if(6===e[0]&&o.label<s[1]){o.label=s[1],s=e;break}if(s&&o.label<s[2]){o.label=s[2],o.ops.push(e);break}s[2]&&o.ops.pop(),o.trys.pop();continue}e=r.call(t,o)}catch(A){e=[6,A],n=0}finally{B=s=0}if(5&e[0])throw e[1];return{value:e[0]?e[1]:void 0,done:!0}}([e,A])}}}var I=(n.prototype.add=function(A,e,t,r){return new n(this.left+A,this.top+e,this.width+t,this.height+r)},n.fromClientRect=function(A){return new n(A.left,A.top,A.width,A.height)},n);function n(A,e,t,r){this.left=A,this.top=e,this.width=t,this.height=r}for(var T=function(A){return I.fromClientRect(A.getBoundingClientRect())},c=function(A){for(var e=[],t=0,r=A.length;t<r;){var B=A.charCodeAt(t++);if(55296<=B&&B<=56319&&t<r){var n=A.charCodeAt(t++);56320==(64512&n)?e.push(((1023&B)<<10)+(1023&n)+65536):(e.push(B),t--)}else e.push(B)}return e},l=function(){for(var A=[],e=0;e<arguments.length;e++)A[e]=arguments[e];if(String.fromCodePoint)return String.fromCodePoint.apply(String,A);var t=A.length;if(!t)return&quot;&quot;;for(var r=[],B=-1,n=&quot;&quot;;++B<t;){var s=A[B];s<=65535?r.push(s):(s-=65536,r.push(55296+(s>>10),s%1024+56320)),(B+1===t||16384<r.length)&&(n+=String.fromCharCode.apply(String,r),r.length=0)}return n},e=&quot;ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/&quot;,Q=&quot;undefined&quot;==typeof Uint8Array?[]:new Uint8Array(256),t=0;t<e.length;t++)Q[e.charCodeAt(t)]=t;function s(A,e,t){return A.slice?A.slice(e,t):new Uint16Array(Array.prototype.slice.call(A,e,t))}var o=(i.prototype.get=function(A){var e;if(0<=A){if(A<55296||56319<A&&A<=65535)return e=((e=this.index[A>>5])<<2)+(31&A),this.data[e];if(A<=65535)return e=((e=this.index[2048+(A-55296>>5)])<<2)+(31&A),this.data[e];if(A<this.highStart)return e=2080+(A>>11),e=this.index[e],e+=A>>5&63,e=((e=this.index[e])<<2)+(31&A),this.data[e];if(A<=1114111)return this.data[this.highValueIndex]}return this.errorValue},i);function i(A,e,t,r,B,n){this.initialValue=A,this.errorValue=e,this.highStart=t,this.highValueIndex=r,this.index=B,this.data=n}function C(A,e,t,r){var B=r[t];if(Array.isArray(A)?-1!==A.indexOf(B):A===B)for(var n=t;n<=r.length;){if((i=r[++n])===e)return!0;if(i!==H)break}if(B===H)for(n=t;0<n;){var s=r[--n];if(Array.isArray(A)?-1!==A.indexOf(s):A===s)for(var o=t;o<=r.length;){var i;if((i=r[++o])===e)return!0;if(i!==H)break}if(s!==H)break}return!1}function g(A,e){for(var t=A;0<=t;){var r=e[t];if(r!==H)return r;t--}return 0}function w(A,e,t,r,B){if(0===t[r])return Y;var n=r-1;if(Array.isArray(B)&&!0===B[n])return Y;var s=n-1,o=1+n,i=e[n],a=0<=s?e[s]:0,c=e[o];if(2===i&&3===c)return Y;if(-1!==j.indexOf(i))return&quot;!&quot;;if(-1!==j.indexOf(c))return Y;if(-1!==$.indexOf(c))return Y;if(8===g(n,e))return&quot;÷&quot;;if(11===q.get(A[n])&&(c===X||c===P||c===x))return Y;if(7===i||7===c)return Y;if(9===i)return Y;if(-1===[H,d,f].indexOf(i)&&9===c)return Y;if(-1!==[p,N,m,v,y].indexOf(c))return Y;if(g(n,e)===O)return Y;if(C(23,O,n,e))return Y;if(C([p,N],L,n,e))return Y;if(C(12,12,n,e))return Y;if(i===H)return&quot;÷&quot;;if(23===i||23===c)return Y;if(16===c||16===i)return&quot;÷&quot;;if(-1!==[d,f,L].indexOf(c)||14===i)return Y;if(36===a&&-1!==rA.indexOf(i))return Y;if(i===y&&36===c)return Y;if(c===R&&-1!==Z.concat(R,m,D,X,P,x).indexOf(i))return Y;if(-1!==Z.indexOf(c)&&i===D||-1!==Z.indexOf(i)&&c===D)return Y;if(i===M&&-1!==[X,P,x].indexOf(c)||-1!==[X,P,x].indexOf(i)&&c===S)return Y;if(-1!==Z.indexOf(i)&&-1!==AA.indexOf(c)||-1!==AA.indexOf(i)&&-1!==Z.indexOf(c))return Y;if(-1!==[M,S].indexOf(i)&&(c===D||-1!==[O,f].indexOf(c)&&e[1+o]===D)||-1!==[O,f].indexOf(i)&&c===D||i===D&&-1!==[D,y,v].indexOf(c))return Y;if(-1!==[D,y,v,p,N].indexOf(c))for(var Q=n;0<=Q;){if((w=e[Q])===D)return Y;if(-1===[y,v].indexOf(w))break;Q--}if(-1!==[M,S].indexOf(c))for(Q=-1!==[p,N].indexOf(i)?s:n;0<=Q;){var w;if((w=e[Q])===D)return Y;if(-1===[y,v].indexOf(w))break;Q--}if(J===i&&-1!==[J,G,V,z].indexOf(c)||-1!==[G,V].indexOf(i)&&-1!==[G,k].indexOf(c)||-1!==[k,z].indexOf(i)&&c===k)return Y;if(-1!==tA.indexOf(i)&&-1!==[R,S].indexOf(c)||-1!==tA.indexOf(c)&&i===M)return Y;if(-1!==Z.indexOf(i)&&-1!==Z.indexOf(c))return Y;if(i===v&&-1!==Z.indexOf(c))return Y;if(-1!==Z.concat(D).indexOf(i)&&c===O||-1!==Z.concat(D).indexOf(c)&&i===N)return Y;if(41===i&&41===c){for(var u=t[n],U=1;0<u&&41===e[--u];)U++;if(U%2!=0)return Y}return i===P&&c===x?Y:&quot;÷&quot;}function u(t,A){A||(A={lineBreak:&quot;normal&quot;,wordBreak:&quot;normal&quot;});var e=function(A,B){void 0===B&&(B=&quot;strict&quot;);var n=[],s=[],o=[];return A.forEach(function(A,e){var t=q.get(A);if(50<t?(o.push(!0),t-=50):o.push(!1),-1!==[&quot;normal&quot;,&quot;auto&quot;,&quot;loose&quot;].indexOf(B)&&-1!==[8208,8211,12316,12448].indexOf(A))return s.push(e),n.push(16);if(4!==t&&11!==t)return s.push(e),31===t?n.push(&quot;strict&quot;===B?L:X):t===W?n.push(_):29===t?n.push(_):43===t?131072<=A&&A<=196605||196608<=A&&A<=262141?n.push(X):n.push(_):void n.push(t);if(0===e)return s.push(e),n.push(_);var r=n[e-1];return-1===eA.indexOf(r)?(s.push(s[e-1]),n.push(r)):(s.push(e),n.push(_))}),[s,n,o]}(t,A.lineBreak),r=e[0],B=e[1],n=e[2];return&quot;break-all&quot;!==A.wordBreak&&&quot;break-word&quot;!==A.wordBreak||(B=B.map(function(A){return-1!==[D,_,W].indexOf(A)?X:A})),[r,B,&quot;keep-all&quot;===A.wordBreak?n.map(function(A,e){return A&&19968<=t[e]&&t[e]<=40959}):void 0]}var a,U,E,F,h,H=10,d=13,f=15,p=17,N=18,m=19,R=20,L=21,O=22,v=24,D=25,S=26,M=27,y=28,_=30,P=32,x=33,V=34,z=35,X=37,J=38,G=39,k=40,W=42,Y=&quot;×&quot;,q=(a=function(A){var e,t,r,B,n,s=.75*A.length,o=A.length,i=0;&quot;=&quot;===A[A.length-1]&&(s--,&quot;=&quot;===A[A.length-2]&&s--);var a=&quot;undefined&quot;!=typeof ArrayBuffer&&&quot;undefined&quot;!=typeof Uint8Array&&void 0!==Uint8Array.prototype.slice?new ArrayBuffer(s):new Array(s),c=Array.isArray(a)?a:new Uint8Array(a);for(e=0;e<o;e+=4)t=Q[A.charCodeAt(e)],r=Q[A.charCodeAt(e+1)],B=Q[A.charCodeAt(e+2)],n=Q[A.charCodeAt(e+3)],c[i++]=t<<2|r>>4,c[i++]=(15&r)<<4|B>>2,c[i++]=(3&B)<<6|63&n;return a}(&quot;KwAAAAAAAAAACA4AIDoAAPAfAAACAAAAAAAIABAAGABAAEgAUABYAF4AZgBeAGYAYABoAHAAeABeAGYAfACEAIAAiACQAJgAoACoAK0AtQC9AMUAXgBmAF4AZgBeAGYAzQDVAF4AZgDRANkA3gDmAOwA9AD8AAQBDAEUARoBIgGAAIgAJwEvATcBPwFFAU0BTAFUAVwBZAFsAXMBewGDATAAiwGTAZsBogGkAawBtAG8AcIBygHSAdoB4AHoAfAB+AH+AQYCDgIWAv4BHgImAi4CNgI+AkUCTQJTAlsCYwJrAnECeQKBAk0CiQKRApkCoQKoArACuALAAsQCzAIwANQC3ALkAjAA7AL0AvwCAQMJAxADGAMwACADJgMuAzYDPgOAAEYDSgNSA1IDUgNaA1oDYANiA2IDgACAAGoDgAByA3YDfgOAAIQDgACKA5IDmgOAAIAAogOqA4AAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAK8DtwOAAIAAvwPHA88D1wPfAyAD5wPsA/QD/AOAAIAABAQMBBIEgAAWBB4EJgQuBDMEIAM7BEEEXgBJBCADUQRZBGEEaQQwADAAcQQ+AXkEgQSJBJEEgACYBIAAoASoBK8EtwQwAL8ExQSAAIAAgACAAIAAgACgAM0EXgBeAF4AXgBeAF4AXgBeANUEXgDZBOEEXgDpBPEE+QQBBQkFEQUZBSEFKQUxBTUFPQVFBUwFVAVcBV4AYwVeAGsFcwV7BYMFiwWSBV4AmgWgBacFXgBeAF4AXgBeAKsFXgCyBbEFugW7BcIFwgXIBcIFwgXQBdQF3AXkBesF8wX7BQMGCwYTBhsGIwYrBjMGOwZeAD8GRwZNBl4AVAZbBl4AXgBeAF4AXgBeAF4AXgBeAF4AXgBeAGMGXgBqBnEGXgBeAF4AXgBeAF4AXgBeAF4AXgB5BoAG4wSGBo4GkwaAAIADHgR5AF4AXgBeAJsGgABGA4AAowarBrMGswagALsGwwbLBjAA0wbaBtoG3QbaBtoG2gbaBtoG2gblBusG8wb7BgMHCwcTBxsHCwcjBysHMAc1BzUHOgdCB9oGSgdSB1oHYAfaBloHaAfaBlIH2gbaBtoG2gbaBtoG2gbaBjUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHbQdeAF4ANQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQd1B30HNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B4MH2gaKB68EgACAAIAAgACAAIAAgACAAI8HlwdeAJ8HpweAAIAArwe3B14AXgC/B8UHygcwANAH2AfgB4AA6AfwBz4B+AcACFwBCAgPCBcIogEYAR8IJwiAAC8INwg/CCADRwhPCFcIXwhnCEoDGgSAAIAAgABvCHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIhAiLCI4IMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlggwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAANQc1BzUHNQc1BzUHNQc1BzUHNQc1B54INQc1B6II2gaqCLIIugiAAIAAvgjGCIAAgACAAIAAgACAAIAAgACAAIAAywiHAYAA0wiAANkI3QjlCO0I9Aj8CIAAgACAAAIJCgkSCRoJIgknCTYHLwk3CZYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiAAIAAAAFAAXgBeAGAAcABeAHwAQACQAKAArQC9AJ4AXgBeAE0A3gBRAN4A7AD8AMwBGgEAAKcBNwEFAUwBXAF4QkhCmEKnArcCgAHHAsABz4LAAcABwAHAAd+C6ABoAG+C/4LAAcABwAHAAc+DF4MAAcAB54M3gweDV4Nng3eDaABoAGgAaABoAGgAaABoAGgAaABoAGgAaABoAGgAaABoAGgAaABoAEeDqABVg6WDqABoQ6gAaABoAHXDvcONw/3DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DncPAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcAB7cPPwlGCU4JMACAAIAAgABWCV4JYQmAAGkJcAl4CXwJgAkwADAAMAAwAIgJgACLCZMJgACZCZ8JowmrCYAAswkwAF4AXgB8AIAAuwkABMMJyQmAAM4JgADVCTAAMAAwADAAgACAAIAAgACAAIAAgACAAIAAqwYWBNkIMAAwADAAMADdCeAJ6AnuCR4E9gkwAP4JBQoNCjAAMACAABUK0wiAAB0KJAosCjQKgAAwADwKQwqAAEsKvQmdCVMKWwowADAAgACAALcEMACAAGMKgABrCjAAMAAwADAAMAAwADAAMAAwADAAMAAeBDAAMAAwADAAMAAwADAAMAAwADAAMAAwAIkEPQFzCnoKiQSCCooKkAqJBJgKoAqkCokEGAGsCrQKvArBCjAAMADJCtEKFQHZCuEK/gHpCvEKMAAwADAAMACAAIwE+QowAIAAPwEBCzAAMAAwADAAMACAAAkLEQswAIAAPwEZCyELgAAOCCkLMAAxCzkLMAAwADAAMAAwADAAXgBeAEELMAAwADAAMAAwADAAMAAwAEkLTQtVC4AAXAtkC4AAiQkwADAAMAAwADAAMAAwADAAbAtxC3kLgAuFC4sLMAAwAJMLlwufCzAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAApwswADAAMACAAIAAgACvC4AAgACAAIAAgACAALcLMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAvwuAAMcLgACAAIAAgACAAIAAyguAAIAAgACAAIAA0QswADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAANkLgACAAIAA4AswADAAMAAwADAAMAAwADAAMAAwADAAMAAwAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACJCR4E6AswADAAhwHwC4AA+AsADAgMEAwwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMACAAIAAGAwdDCUMMAAwAC0MNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQw1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHPQwwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADUHNQc1BzUHNQc1BzUHNQc2BzAAMAA5DDUHNQc1BzUHNQc1BzUHNQc1BzUHNQdFDDAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAgACAAIAATQxSDFoMMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAF4AXgBeAF4AXgBeAF4AYgxeAGoMXgBxDHkMfwxeAIUMXgBeAI0MMAAwADAAMAAwAF4AXgCVDJ0MMAAwADAAMABeAF4ApQxeAKsMswy7DF4Awgy9DMoMXgBeAF4AXgBeAF4AXgBeAF4AXgDRDNkMeQBqCeAM3Ax8AOYM7Az0DPgMXgBeAF4AXgBeAF4AXgBeAF4AXgBeAF4AXgBeAF4AXgCgAAANoAAHDQ4NFg0wADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAeDSYNMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAIAAgACAAIAAgACAAC4NMABeAF4ANg0wADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAD4NRg1ODVYNXg1mDTAAbQ0wADAAMAAwADAAMAAwADAA2gbaBtoG2gbaBtoG2gbaBnUNeg3CBYANwgWFDdoGjA3aBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gaUDZwNpA2oDdoG2gawDbcNvw3HDdoG2gbPDdYN3A3fDeYN2gbsDfMN2gbaBvoN/g3aBgYODg7aBl4AXgBeABYOXgBeACUG2gYeDl4AJA5eACwO2w3aBtoGMQ45DtoG2gbaBtoGQQ7aBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gZJDjUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B1EO2gY1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQdZDjUHNQc1BzUHNQc1B2EONQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHaA41BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B3AO2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gY1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B2EO2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gZJDtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBkkOeA6gAKAAoAAwADAAMAAwAKAAoACgAKAAoACgAKAAgA4wADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAD//wQABAAEAAQABAAEAAQABAAEAA0AAwABAAEAAgAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAKABMAFwAeABsAGgAeABcAFgASAB4AGwAYAA8AGAAcAEsASwBLAEsASwBLAEsASwBLAEsAGAAYAB4AHgAeABMAHgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAFgAbABIAHgAeAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQABYADQARAB4ABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAUABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAkAFgAaABsAGwAbAB4AHQAdAB4ATwAXAB4ADQAeAB4AGgAbAE8ATwAOAFAAHQAdAB0ATwBPABcATwBPAE8AFgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAB4AHgAeAB4AUABQAFAAUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AHgAeAFAATwBAAE8ATwBPAEAATwBQAFAATwBQAB4AHgAeAB4AHgAeAB0AHQAdAB0AHgAdAB4ADgBQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgBQAB4AUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAJAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAkACQAJAAkACQAJAAkABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAeAB4AHgAeAFAAHgAeAB4AKwArAFAAUABQAFAAGABQACsAKwArACsAHgAeAFAAHgBQAFAAUAArAFAAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAEAAQABAAEAAQABAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAUAAeAB4AHgAeAB4AHgArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwAYAA0AKwArAB4AHgAbACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQADQAEAB4ABAAEAB4ABAAEABMABAArACsAKwArACsAKwArACsAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAKwArACsAKwArAFYAVgBWAB4AHgArACsAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AGgAaABoAGAAYAB4AHgAEAAQABAAEAAQABAAEAAQABAAEAAQAEwAEACsAEwATAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABABLAEsASwBLAEsASwBLAEsASwBLABoAGQAZAB4AUABQAAQAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQABMAUAAEAAQABAAEAAQABAAEAB4AHgAEAAQABAAEAAQABABQAFAABAAEAB4ABAAEAAQABABQAFAASwBLAEsASwBLAEsASwBLAEsASwBQAFAAUAAeAB4AUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwAeAFAABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQAUABQAB4AHgAYABMAUAArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAFAABAAEAAQABAAEAFAABAAEAAQAUAAEAAQABAAEAAQAKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAArACsAHgArAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAeAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAFAABAAEAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAAQABAANAA0ASwBLAEsASwBLAEsASwBLAEsASwAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQAKwBQAFAAUABQAFAAUABQAFAAKwArAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAKwArACsAUABQAFAAUAArACsABABQAAQABAAEAAQABAAEAAQAKwArAAQABAArACsABAAEAAQAUAArACsAKwArACsAKwArACsABAArACsAKwArAFAAUAArAFAAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwBQAFAAGgAaAFAAUABQAFAAUABMAB4AGwBQAB4AKwArACsABAAEAAQAKwBQAFAAUABQAFAAUAArACsAKwArAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAUAArAFAAUAArAFAAUAArACsABAArAAQABAAEAAQABAArACsAKwArAAQABAArACsABAAEAAQAKwArACsABAArACsAKwArACsAKwArAFAAUABQAFAAKwBQACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwAEAAQAUABQAFAABAArACsAKwArACsAKwArACsAKwArACsABAAEAAQAKwBQAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAUAArAFAAUABQAFAAUAArACsABABQAAQABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQAKwArAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwAeABsAKwArACsAKwArACsAKwBQAAQABAAEAAQABAAEACsABAAEAAQAKwBQAFAAUABQAFAAUABQAFAAKwArAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQAKwArAAQABAArACsABAAEAAQAKwArACsAKwArACsAKwArAAQABAArACsAKwArAFAAUAArAFAAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwAeAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwAEAFAAKwBQAFAAUABQAFAAUAArACsAKwBQAFAAUAArAFAAUABQAFAAKwArACsAUABQACsAUAArAFAAUAArACsAKwBQAFAAKwArACsAUABQAFAAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwAEAAQABAAEAAQAKwArACsABAAEAAQAKwAEAAQABAAEACsAKwBQACsAKwArACsAKwArAAQAKwArACsAKwArACsAKwArACsAKwBLAEsASwBLAEsASwBLAEsASwBLAFAAUABQAB4AHgAeAB4AHgAeABsAHgArACsAKwArACsABAAEAAQABAArAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArAFAABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQABAArACsAKwArACsAKwArAAQABAArAFAAUABQACsAKwArACsAKwBQAFAABAAEACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAB4AUAAEAAQABAArAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQACsAKwAEAFAABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQABAArACsAKwArACsAKwArAAQABAArACsAKwArACsAKwArAFAAKwBQAFAABAAEACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAFAABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQABABQAB4AKwArACsAKwBQAFAAUAAEAFAAUABQAFAAUABQAFAAUABQAFAABAAEACsAKwBLAEsASwBLAEsASwBLAEsASwBLAFAAUABQAFAAUABQAFAAUABQABoAUABQAFAAUABQAFAAKwArAAQABAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQACsAUAArACsAUABQAFAAUABQAFAAUAArACsAKwAEACsAKwArACsABAAEAAQABAAEAAQAKwAEACsABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArAAQABAAeACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAAqAFwAXAAqACoAKgAqACoAKgAqACsAKwArACsAGwBcAFwAXABcAFwAXABcACoAKgAqACoAKgAqACoAKgAeAEsASwBLAEsASwBLAEsASwBLAEsADQANACsAKwArACsAKwBcAFwAKwBcACsAKwBcAFwAKwBcACsAKwBcACsAKwArACsAKwArAFwAXABcAFwAKwBcAFwAXABcAFwAXABcACsAXABcAFwAKwBcACsAXAArACsAXABcACsAXABcAFwAXAAqAFwAXAAqACoAKgAqACoAKgArACoAKgBcACsAKwBcAFwAXABcAFwAKwBcACsAKgAqACoAKgAqACoAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArAFwAXABcAFwAUAAOAA4ADgAOAB4ADgAOAAkADgAOAA0ACQATABMAEwATABMACQAeABMAHgAeAB4ABAAEAB4AHgAeAB4AHgAeAEsASwBLAEsASwBLAEsASwBLAEsAUABQAFAAUABQAFAAUABQAFAAUAANAAQAHgAEAB4ABAAWABEAFgARAAQABABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAANAAQABAAEAAQABAANAAQABABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsADQANAB4AHgAeAB4AHgAeAAQAHgAeAB4AHgAeAB4AKwAeAB4ADgAOAA0ADgAeAB4AHgAeAB4ACQAJACsAKwArACsAKwBcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqAFwASwBLAEsASwBLAEsASwBLAEsASwANAA0AHgAeAB4AHgBcAFwAXABcAFwAXAAqACoAKgAqAFwAXABcAFwAKgAqACoAXAAqACoAKgBcAFwAKgAqACoAKgAqACoAKgBcAFwAXAAqACoAKgAqAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAKgAqACoAKgAqACoAKgAqACoAXAAqAEsASwBLAEsASwBLAEsASwBLAEsAKgAqACoAKgAqACoAUABQAFAAUABQAFAAKwBQACsAKwArACsAKwBQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAFAAUABQAFAAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQACsAKwBQAFAAUABQAFAAUABQACsAUAArAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUAArACsAUABQAFAAUABQAFAAUAArAFAAKwBQAFAAUABQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwAEAAQABAAeAA0AHgAeAB4AHgAeAB4AHgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AHgAeAB4AHgAeAB4AHgAeACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQACsAKwANAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAA0AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQABYAEQArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAADQANAA0AUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAABAAEAAQAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAA0ADQArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsABAAEACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoADQANABUAXAANAB4ADQAbAFwAKgArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArAB4AHgATABMADQANAA4AHgATABMAHgAEAAQABAAJACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAUABQAFAAUABQAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABABQACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwArACsAKwAeACsAKwArABMAEwBLAEsASwBLAEsASwBLAEsASwBLAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcACsAKwBcAFwAXABcAFwAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcACsAKwArACsAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwBcACsAKwArACoAKgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEACsAKwAeAB4AXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAKgAqACoAKgAqACoAKgArACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgArACsABABLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAKgAqACoAKgAqACoAKgBcACoAKgAqACoAKgAqACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQAUABQAFAAUABQAFAAUAArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsADQANAB4ADQANAA0ADQAeAB4AHgAeAB4AHgAeAB4AHgAeAAQABAAEAAQABAAEAAQABAAEAB4AHgAeAB4AHgAeAB4AHgAeACsAKwArAAQABAAEAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAUABQAEsASwBLAEsASwBLAEsASwBLAEsAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwArACsAKwArACsAHgAeAB4AHgBQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwANAA0ADQANAA0ASwBLAEsASwBLAEsASwBLAEsASwArACsAKwBQAFAAUABLAEsASwBLAEsASwBLAEsASwBLAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAANAA0AUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAeAB4AHgAeAB4AHgArACsAKwArACsAKwArACsABAAEAAQAHgAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAFAAUABQAFAABABQAFAAUABQAAQABAAEAFAAUAAEAAQABAArACsAKwArACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwAEAAQABAAEAAQAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUAArAFAAKwBQACsAUAArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAHgAeAB4AHgAeAB4AHgAeAFAAHgAeAB4AUABQAFAAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAKwArAB4AHgAeAB4AHgAeACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAUABQAFAAKwAeAB4AHgAeAB4AHgAeAA4AHgArAA0ADQANAA0ADQANAA0ACQANAA0ADQAIAAQACwAEAAQADQAJAA0ADQAMAB0AHQAeABcAFwAWABcAFwAXABYAFwAdAB0AHgAeABQAFAAUAA0AAQABAAQABAAEAAQABAAJABoAGgAaABoAGgAaABoAGgAeABcAFwAdABUAFQAeAB4AHgAeAB4AHgAYABYAEQAVABUAFQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgANAB4ADQANAA0ADQAeAA0ADQANAAcAHgAeAB4AHgArAAQABAAEAAQABAAEAAQABAAEAAQAUABQACsAKwBPAFAAUABQAFAAUAAeAB4AHgAWABEATwBQAE8ATwBPAE8AUABQAFAAUABQAB4AHgAeABYAEQArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAGwAbABsAGwAbABsAGwAaABsAGwAbABsAGwAbABsAGwAbABsAGwAbABsAGwAaABsAGwAbABsAGgAbABsAGgAbABsAGwAbABsAGwAbABsAGwAbABsAGwAbABsAGwAbABsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgBQABoAHgAdAB4AUAAeABoAHgAeAB4AHgAeAB4AHgAeAB4ATwAeAFAAGwAeAB4AUABQAFAAUABQAB4AHgAeAB0AHQAeAFAAHgBQAB4AUAAeAFAATwBQAFAAHgAeAB4AHgAeAB4AHgBQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAB4AUABQAFAAUABPAE8AUABQAFAAUABQAE8AUABQAE8AUABPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBQAFAAUABQAE8ATwBPAE8ATwBPAE8ATwBPAE8AUABQAFAAUABQAFAAUABQAFAAHgAeAFAAUABQAFAATwAeAB4AKwArACsAKwAdAB0AHQAdAB0AHQAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAeAB0AHQAeAB4AHgAdAB0AHgAeAB0AHgAeAB4AHQAeAB0AGwAbAB4AHQAeAB4AHgAeAB0AHgAeAB0AHQAdAB0AHgAeAB0AHgAdAB4AHQAdAB0AHQAdAB0AHgAdAB4AHgAeAB4AHgAdAB0AHQAdAB4AHgAeAB4AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAeAB4AHgAdAB4AHgAeAB4AHgAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB4AHgAdAB0AHQAdAB4AHgAdAB0AHgAeAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB0AHgAeAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHgAeAB4AHQAeAB4AHgAeAB4AHgAeAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeABQAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAWABEAFgARAB4AHgAeAB4AHgAeAB0AHgAeAB4AHgAeAB4AHgAlACUAHgAeAB4AHgAeAB4AHgAeAB4AFgARAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBQAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB4AHgAeAB4AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHgAeAB0AHQAdAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB0AHgAdAB0AHQAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAdAB0AHgAeAB0AHQAeAB4AHgAeAB0AHQAeAB4AHgAeAB0AHQAdAB4AHgAdAB4AHgAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAeAB0AHQAeAB4AHQAeAB4AHgAeAB0AHQAeAB4AHgAeACUAJQAdAB0AJQAeACUAJQAlACAAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAHgAeAB4AHgAdAB4AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB4AHQAdAB0AHgAdACUAHQAdAB4AHQAdAB4AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlACUAJQAlACUAHQAdAB0AHQAlAB4AJQAlACUAHQAlACUAHQAdAB0AJQAlAB0AHQAlAB0AHQAlACUAJQAeAB0AHgAeAB4AHgAdAB0AJQAdAB0AHQAdAB0AHQAlACUAJQAlACUAHQAlACUAIAAlAB0AHQAlACUAJQAlACUAJQAlACUAHgAeAB4AJQAlACAAIAAgACAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHgAeABcAFwAXABcAFwAXAB4AEwATACUAHgAeAB4AFgARABYAEQAWABEAFgARABYAEQAWABEAFgARAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAWABEAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AFgARABYAEQAWABEAFgARABYAEQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeABYAEQAWABEAFgARABYAEQAWABEAFgARABYAEQAWABEAFgARABYAEQAWABEAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AFgARABYAEQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeABYAEQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwAeAB4AHgAeAB4AHgAeAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAEAAQABAAeAB4AKwArACsAKwArABMADQANAA0AUAATAA0AUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAUAANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAEAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQACsAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAA0ADQANAA0ADQANAA0ADQAeAA0AFgANAB4AHgAXABcAHgAeABcAFwAWABEAFgARABYAEQAWABEADQANAA0ADQATAFAADQANAB4ADQANAB4AHgAeAB4AHgAMAAwADQANAA0AHgANAA0AFgANAA0ADQANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAKwArACsAKwArACsAKwArACsAKwArACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAlACUAJQAlACUAJQAlACUAJQAlACUAJQArACsAKwArAA0AEQARACUAJQBHAFcAVwAWABEAFgARABYAEQAWABEAFgARACUAJQAWABEAFgARABYAEQAWABEAFQAWABEAEQAlAFcAVwBXAFcAVwBXAFcAVwBXAAQABAAEAAQABAAEACUAVwBXAFcAVwA2ACUAJQBXAFcAVwBHAEcAJQAlACUAKwBRAFcAUQBXAFEAVwBRAFcAUQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFEAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBRAFcAUQBXAFEAVwBXAFcAVwBXAFcAUQBXAFcAVwBXAFcAVwBRAFEAKwArAAQABAAVABUARwBHAFcAFQBRAFcAUQBXAFEAVwBRAFcAUQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFEAVwBRAFcAUQBXAFcAVwBXAFcAVwBRAFcAVwBXAFcAVwBXAFEAUQBXAFcAVwBXABUAUQBHAEcAVwArACsAKwArACsAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwArACUAJQBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArACsAKwArACUAJQAlACUAKwArACsAKwArACsAKwArACsAKwArACsAUQBRAFEAUQBRAFEAUQBRAFEAUQBRAFEAUQBRAFEAUQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACsAVwBXAFcAVwBXAFcAVwBXAFcAVwAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAE8ATwBPAE8ATwBPAE8ATwAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAEcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAADQATAA0AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABLAEsASwBLAEsASwBLAEsASwBLAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAABAAEAAQABAAeAAQABAAEAAQABAAEAAQABAAEAAQAHgBQAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AUABQAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAeAA0ADQANAA0ADQArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAB4AHgAeAB4AHgAeAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAB4AHgAeAB4AHgAeAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAAQAUABQAFAABABQAFAAUABQAAQAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAeAB4AHgAeACsAKwArACsAUABQAFAAUABQAFAAHgAeABoAHgArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAADgAOABMAEwArACsAKwArACsAKwArACsABAAEAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwANAA0ASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABABQAFAAUABQAFAAUAAeAB4AHgBQAA4AUAArACsAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAA0ADQBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwArACsAKwArACsAKwArACsAKwArAB4AWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYACsAKwArAAQAHgAeAB4AHgAeAB4ADQANAA0AHgAeAB4AHgArAFAASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArAB4AHgBcAFwAXABcAFwAKgBcAFwAXABcAFwAXABcAFwAXABcAEsASwBLAEsASwBLAEsASwBLAEsAXABcAFwAXABcACsAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArAFAAUABQAAQAUABQAFAAUABQAFAAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAHgANAA0ADQBcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAXAAqACoAKgBcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAAqAFwAKgAqACoAXABcACoAKgBcAFwAXABcAFwAKgAqAFwAKgBcACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcACoAKgBQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAA0ADQBQAFAAUAAEAAQAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUAArACsAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQADQAEAAQAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAVABVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBUAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVACsAKwArACsAKwArACsAKwArACsAKwArAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAKwArACsAKwBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAKwArACsAKwAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAKwArACsAKwArAFYABABWAFYAVgBWAFYAVgBWAFYAVgBWAB4AVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgArAFYAVgBWAFYAVgArAFYAKwBWAFYAKwBWAFYAKwBWAFYAVgBWAFYAVgBWAFYAVgBWAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAEQAWAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUAAaAB4AKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAGAARABEAGAAYABMAEwAWABEAFAArACsAKwArACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACUAJQAlACUAJQAWABEAFgARABYAEQAWABEAFgARABYAEQAlACUAFgARACUAJQAlACUAJQAlACUAEQAlABEAKwAVABUAEwATACUAFgARABYAEQAWABEAJQAlACUAJQAlACUAJQAlACsAJQAbABoAJQArACsAKwArAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAAcAKwATACUAJQAbABoAJQAlABYAEQAlACUAEQAlABEAJQBXAFcAVwBXAFcAVwBXAFcAVwBXABUAFQAlACUAJQATACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXABYAJQARACUAJQAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwAWACUAEQAlABYAEQARABYAEQARABUAVwBRAFEAUQBRAFEAUQBRAFEAUQBRAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAEcARwArACsAVwBXAFcAVwBXAFcAKwArAFcAVwBXAFcAVwBXACsAKwBXAFcAVwBXAFcAVwArACsAVwBXAFcAKwArACsAGgAbACUAJQAlABsAGwArAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwAEAAQABAAQAB0AKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsADQANAA0AKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAFAAHgAeAB4AKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAKwArAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAAQAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsADQBQAFAAUABQACsAKwArACsAUABQAFAAUABQAFAAUABQAA0AUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUAArACsAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQACsAKwArAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAA0AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAB4AHgBQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsADQBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwBQAFAAUABQAFAABAAEAAQAKwAEAAQAKwArACsAKwArAAQABAAEAAQAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsABAAEAAQAKwArACsAKwAEAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsADQANAA0ADQANAA0ADQANAB4AKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAB4AUABQAFAAUABQAFAAUABQAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEACsAKwArACsAUABQAFAAUABQAA0ADQANAA0ADQANABQAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwANAA0ADQANAA0ADQANAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAHgAeAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwBQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAA0ADQAeAB4AHgAeAB4AKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAeAB4AHgANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAKwArAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsASwBLAEsASwBLAEsASwBLAEsASwANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAeAA4AUAArACsAKwArACsAKwArACsAKwAEAFAAUABQAFAADQANAB4ADQAeAAQABAAEAB4AKwArAEsASwBLAEsASwBLAEsASwBLAEsAUAAOAFAADQANAA0AKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAANAA0AHgANAA0AHgAEACsAUABQAFAAUABQAFAAUAArAFAAKwBQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAA0AKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsABAAEAAQABAArAFAAUABQAFAAUABQAFAAUAArACsAUABQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAArACsABAAEACsAKwAEAAQABAArACsAUAArACsAKwArACsAKwAEACsAKwArACsAKwBQAFAAUABQAFAABAAEACsAKwAEAAQABAAEAAQABAAEACsAKwArAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAQABABQAFAAUABQAA0ADQANAA0AHgBLAEsASwBLAEsASwBLAEsASwBLACsADQArAB4AKwArAAQABAAEAAQAUABQAB4AUAArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEACsAKwAEAAQABAAEAAQABAAEAAQABAAOAA0ADQATABMAHgAeAB4ADQANAA0ADQANAA0ADQANAA0ADQANAA0ADQANAA0AUABQAFAAUAAEAAQAKwArAAQADQANAB4AUAArACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwAOAA4ADgAOAA4ADgAOAA4ADgAOAA4ADgAOACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAArACsAKwAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAXABcAA0ADQANACoASwBLAEsASwBLAEsASwBLAEsASwBQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwBQAFAABAAEAAQABAAEAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAFAABAAEAAQABAAOAB4ADQANAA0ADQAOAB4ABAArACsAKwArACsAKwArACsAUAAEAAQABAAEAAQABAAEAAQABAAEAAQAUABQAFAAUAArACsAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAA0ADQANACsADgAOAA4ADQANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEACsABAAEAAQABAAEAAQABAAEAFAADQANAA0ADQANACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwAOABMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQACsAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAArACsAKwAEACsABAAEACsABAAEAAQABAAEAAQABABQAAQAKwArACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsADQANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAASABIAEgAQwBDAEMAUABQAFAAUABDAFAAUABQAEgAQwBIAEMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAASABDAEMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABIAEMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwANAA0AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAAQABAAEAAQABAANACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAA0ADQANAB4AHgAeAB4AHgAeAFAAUABQAFAADQAeACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAEcARwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwArACsAKwArACsAKwArACsAKwArACsAKwArAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQACsAKwAeAAQABAANAAQABAAEAAQAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAEAAQABAAEAB4AHgAeAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAHgAeAAQABAAEAAQABAAEAAQAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAEAAQABAAEAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAEAAQABAAeACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAFAAUAArACsAUAArACsAUABQACsAKwBQAFAAUABQACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwBQACsAUABQAFAAUABQAFAAUAArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAKwAeAB4AUABQAFAAUABQACsAUAArACsAKwBQAFAAUABQAFAAUABQACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AKwArAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAB4AHgAeAB4ABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAB4AHgAeAB4AHgAeAB4AHgAEAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAeAB4ADQANAA0ADQAeACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAAQABAAEAAQABAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAQABAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsABAAEAAQABAAEAAQABAArAAQABAArAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAEAAQABAAEAAQABAAEACsAKwArACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAFgAWAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUAArAFAAKwArAFAAKwBQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUAArAFAAKwBQACsAKwArACsAKwArAFAAKwArACsAKwBQACsAUAArAFAAKwBQAFAAUAArAFAAUAArAFAAKwArAFAAKwBQACsAUAArAFAAKwBQACsAUABQACsAUAArACsAUABQAFAAUAArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQACsAUABQAFAAUAArAFAAKwBQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwBQAFAAUAArAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwArACsATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwAlACUAJQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAeACUAHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHgAeACUAJQAlACUAHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQAlACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAHgAeACUAJQAlACUAJQAeACUAJQAlACUAJQAgACAAIAAlACUAIAAlACUAIAAgACAAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIQAhACEAIQAhACUAJQAgACAAJQAlACAAIAAgACAAIAAgACAAIAAgACAAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAgACAAIAAlACUAJQAlACAAJQAgACAAIAAgACAAIAAgACAAIAAlACUAJQAgACUAJQAlACUAIAAgACAAJQAgACAAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeACUAHgAlAB4AJQAlACUAJQAlACAAJQAlACUAJQAeACUAHgAeACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAHgAeAB4AHgAeAB4AHgAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAgACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAlACUAJQAlACAAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAIAAgACAAJQAlACUAIAAgACAAIAAgAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AFwAXABcAFQAVABUAHgAeAB4AHgAlACUAJQAgACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAgACAAJQAlACUAJQAlACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAJQAlACUAJQAlACAAIAAgACAAIAAlACAAIAAlACUAJQAlACUAJQAgACUAJQAlACUAJQAlACUAJQAlACAAIAAgACAAIAAgACAAIAAgACAAJQAlACUAIAAgACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACsAKwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAJQAlACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAJQAlACUAJQAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQArAAQAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsA&quot;),U=Array.isArray(a)?function(A){for(var e=A.length,t=[],r=0;r<e;r+=4)t.push(A[r+3]<<24|A[r+2]<<16|A[r+1]<<8|A[r]);return t}(a):new Uint32Array(a),E=Array.isArray(a)?function(A){for(var e=A.length,t=[],r=0;r<e;r+=2)t.push(A[r+1]<<8|A[r]);return t}(a):new Uint16Array(a),F=s(E,12,U[4]/2),h=2===U[5]?s(E,(24+U[4])/2):function(A,e,t){return A.slice?A.slice(e,t):new Uint32Array(Array.prototype.slice.call(A,e,t))}(U,Math.ceil((24+U[4])/4)),new o(U[0],U[1],U[2],U[3],F,h)),Z=[_,36],j=[1,2,3,5],$=[H,8],AA=[M,S],eA=j.concat($),tA=[J,G,k,V,z],rA=[f,d],BA=(nA.prototype.slice=function(){return l.apply(void 0,this.codePoints.slice(this.start,this.end))},nA);function nA(A,e,t,r){this.codePoints=A,this.required=&quot;!&quot;===e,this.start=t,this.end=r}var sA,oA;(oA=sA||(sA={}))[oA.STRING_TOKEN=0]=&quot;STRING_TOKEN&quot;,oA[oA.BAD_STRING_TOKEN=1]=&quot;BAD_STRING_TOKEN&quot;,oA[oA.LEFT_PARENTHESIS_TOKEN=2]=&quot;LEFT_PARENTHESIS_TOKEN&quot;,oA[oA.RIGHT_PARENTHESIS_TOKEN=3]=&quot;RIGHT_PARENTHESIS_TOKEN&quot;,oA[oA.COMMA_TOKEN=4]=&quot;COMMA_TOKEN&quot;,oA[oA.HASH_TOKEN=5]=&quot;HASH_TOKEN&quot;,oA[oA.DELIM_TOKEN=6]=&quot;DELIM_TOKEN&quot;,oA[oA.AT_KEYWORD_TOKEN=7]=&quot;AT_KEYWORD_TOKEN&quot;,oA[oA.PREFIX_MATCH_TOKEN=8]=&quot;PREFIX_MATCH_TOKEN&quot;,oA[oA.DASH_MATCH_TOKEN=9]=&quot;DASH_MATCH_TOKEN&quot;,oA[oA.INCLUDE_MATCH_TOKEN=10]=&quot;INCLUDE_MATCH_TOKEN&quot;,oA[oA.LEFT_CURLY_BRACKET_TOKEN=11]=&quot;LEFT_CURLY_BRACKET_TOKEN&quot;,oA[oA.RIGHT_CURLY_BRACKET_TOKEN=12]=&quot;RIGHT_CURLY_BRACKET_TOKEN&quot;,oA[oA.SUFFIX_MATCH_TOKEN=13]=&quot;SUFFIX_MATCH_TOKEN&quot;,oA[oA.SUBSTRING_MATCH_TOKEN=14]=&quot;SUBSTRING_MATCH_TOKEN&quot;,oA[oA.DIMENSION_TOKEN=15]=&quot;DIMENSION_TOKEN&quot;,oA[oA.PERCENTAGE_TOKEN=16]=&quot;PERCENTAGE_TOKEN&quot;,oA[oA.NUMBER_TOKEN=17]=&quot;NUMBER_TOKEN&quot;,oA[oA.FUNCTION=18]=&quot;FUNCTION&quot;,oA[oA.FUNCTION_TOKEN=19]=&quot;FUNCTION_TOKEN&quot;,oA[oA.IDENT_TOKEN=20]=&quot;IDENT_TOKEN&quot;,oA[oA.COLUMN_TOKEN=21]=&quot;COLUMN_TOKEN&quot;,oA[oA.URL_TOKEN=22]=&quot;URL_TOKEN&quot;,oA[oA.BAD_URL_TOKEN=23]=&quot;BAD_URL_TOKEN&quot;,oA[oA.CDC_TOKEN=24]=&quot;CDC_TOKEN&quot;,oA[oA.CDO_TOKEN=25]=&quot;CDO_TOKEN&quot;,oA[oA.COLON_TOKEN=26]=&quot;COLON_TOKEN&quot;,oA[oA.SEMICOLON_TOKEN=27]=&quot;SEMICOLON_TOKEN&quot;,oA[oA.LEFT_SQUARE_BRACKET_TOKEN=28]=&quot;LEFT_SQUARE_BRACKET_TOKEN&quot;,oA[oA.RIGHT_SQUARE_BRACKET_TOKEN=29]=&quot;RIGHT_SQUARE_BRACKET_TOKEN&quot;,oA[oA.UNICODE_RANGE_TOKEN=30]=&quot;UNICODE_RANGE_TOKEN&quot;,oA[oA.WHITESPACE_TOKEN=31]=&quot;WHITESPACE_TOKEN&quot;,oA[oA.EOF_TOKEN=32]=&quot;EOF_TOKEN&quot;;function iA(A){return 48<=A&&A<=57}function aA(A){return iA(A)||65<=A&&A<=70||97<=A&&A<=102}function cA(A){return 10===A||9===A||32===A}function QA(A){return function(A){return function(A){return 97<=A&&A<=122}(A)||function(A){return 65<=A&&A<=90}(A)}(A)||function(A){return 128<=A}(A)||95===A}function wA(A){return QA(A)||iA(A)||45===A}function uA(A,e){return 92===A&&10!==e}function UA(A,e,t){return 45===A?QA(e)||uA(e,t):!!QA(A)||!(92!==A||!uA(A,e))}function lA(A,e,t){return 43===A||45===A?!!iA(e)||46===e&&iA(t):iA(46===A?e:A)}var CA={type:sA.LEFT_PARENTHESIS_TOKEN},gA={type:sA.RIGHT_PARENTHESIS_TOKEN},EA={type:sA.COMMA_TOKEN},FA={type:sA.SUFFIX_MATCH_TOKEN},hA={type:sA.PREFIX_MATCH_TOKEN},HA={type:sA.COLUMN_TOKEN},dA={type:sA.DASH_MATCH_TOKEN},fA={type:sA.INCLUDE_MATCH_TOKEN},pA={type:sA.LEFT_CURLY_BRACKET_TOKEN},NA={type:sA.RIGHT_CURLY_BRACKET_TOKEN},KA={type:sA.SUBSTRING_MATCH_TOKEN},IA={type:sA.BAD_URL_TOKEN},TA={type:sA.BAD_STRING_TOKEN},mA={type:sA.CDO_TOKEN},RA={type:sA.CDC_TOKEN},LA={type:sA.COLON_TOKEN},OA={type:sA.SEMICOLON_TOKEN},vA={type:sA.LEFT_SQUARE_BRACKET_TOKEN},DA={type:sA.RIGHT_SQUARE_BRACKET_TOKEN},SA={type:sA.WHITESPACE_TOKEN},bA={type:sA.EOF_TOKEN},MA=(yA.prototype.write=function(A){this._value=this._value.concat(c(A))},yA.prototype.read=function(){for(var A=[],e=this.consumeToken();e!==bA;)A.push(e),e=this.consumeToken();return A},yA.prototype.consumeToken=function(){var A=this.consumeCodePoint();switch(A){case 34:return this.consumeStringToken(34);case 35:var e=this.peekCodePoint(0),t=this.peekCodePoint(1),r=this.peekCodePoint(2);if(wA(e)||uA(t,r)){var B=UA(e,t,r)?2:1,n=this.consumeName();return{type:sA.HASH_TOKEN,value:n,flags:B}}break;case 36:if(61===this.peekCodePoint(0))return this.consumeCodePoint(),FA;break;case 39:return this.consumeStringToken(39);case 40:return CA;case 41:return gA;case 42:if(61===this.peekCodePoint(0))return this.consumeCodePoint(),KA;break;case 43:if(lA(A,this.peekCodePoint(0),this.peekCodePoint(1)))return this.reconsumeCodePoint(A),this.consumeNumericToken();break;case 44:return EA;case 45:var s=A,o=this.peekCodePoint(0),i=this.peekCodePoint(1);if(lA(s,o,i))return this.reconsumeCodePoint(A),this.consumeNumericToken();if(UA(s,o,i))return this.reconsumeCodePoint(A),this.consumeIdentLikeToken();if(45===o&&62===i)return this.consumeCodePoint(),this.consumeCodePoint(),RA;break;case 46:if(lA(A,this.peekCodePoint(0),this.peekCodePoint(1)))return this.reconsumeCodePoint(A),this.consumeNumericToken();break;case 47:if(42===this.peekCodePoint(0))for(this.consumeCodePoint();;){var a=this.consumeCodePoint();if(42===a&&47===(a=this.consumeCodePoint()))return this.consumeToken();if(-1===a)return this.consumeToken()}break;case 58:return LA;case 59:return OA;case 60:if(33===this.peekCodePoint(0)&&45===this.peekCodePoint(1)&&45===this.peekCodePoint(2))return this.consumeCodePoint(),this.consumeCodePoint(),mA;break;case 64:var c=this.peekCodePoint(0),Q=this.peekCodePoint(1),w=this.peekCodePoint(2);if(UA(c,Q,w))return n=this.consumeName(),{type:sA.AT_KEYWORD_TOKEN,value:n};break;case 91:return vA;case 92:if(uA(A,this.peekCodePoint(0)))return this.reconsumeCodePoint(A),this.consumeIdentLikeToken();break;case 93:return DA;case 61:if(61===this.peekCodePoint(0))return this.consumeCodePoint(),hA;break;case 123:return pA;case 125:return NA;case 117:case 85:var u=this.peekCodePoint(0),U=this.peekCodePoint(1);return 43!==u||!aA(U)&&63!==U||(this.consumeCodePoint(),this.consumeUnicodeRangeToken()),this.reconsumeCodePoint(A),this.consumeIdentLikeToken();case 124:if(61===this.peekCodePoint(0))return this.consumeCodePoint(),dA;if(124===this.peekCodePoint(0))return this.consumeCodePoint(),HA;break;case 126:if(61===this.peekCodePoint(0))return this.consumeCodePoint(),fA;break;case-1:return bA}return cA(A)?(this.consumeWhiteSpace(),SA):iA(A)?(this.reconsumeCodePoint(A),this.consumeNumericToken()):QA(A)?(this.reconsumeCodePoint(A),this.consumeIdentLikeToken()):{type:sA.DELIM_TOKEN,value:l(A)}},yA.prototype.consumeCodePoint=function(){var A=this._value.shift();return void 0===A?-1:A},yA.prototype.reconsumeCodePoint=function(A){this._value.unshift(A)},yA.prototype.peekCodePoint=function(A){return A>=this._value.length?-1:this._value[A]},yA.prototype.consumeUnicodeRangeToken=function(){for(var A=[],e=this.consumeCodePoint();aA(e)&&A.length<6;)A.push(e),e=this.consumeCodePoint();for(var t=!1;63===e&&A.length<6;)A.push(e),e=this.consumeCodePoint(),t=!0;if(t){var r=parseInt(l.apply(void 0,A.map(function(A){return 63===A?48:A})),16),B=parseInt(l.apply(void 0,A.map(function(A){return 63===A?70:A})),16);return{type:sA.UNICODE_RANGE_TOKEN,start:r,end:B}}var n=parseInt(l.apply(void 0,A),16);if(45===this.peekCodePoint(0)&&aA(this.peekCodePoint(1))){this.consumeCodePoint(),e=this.consumeCodePoint();for(var s=[];aA(e)&&s.length<6;)s.push(e),e=this.consumeCodePoint();return B=parseInt(l.apply(void 0,s),16),{type:sA.UNICODE_RANGE_TOKEN,start:n,end:B}}return{type:sA.UNICODE_RANGE_TOKEN,start:n,end:n}},yA.prototype.consumeIdentLikeToken=function(){var A=this.consumeName();return&quot;url&quot;===A.toLowerCase()&&40===this.peekCodePoint(0)?(this.consumeCodePoint(),this.consumeUrlToken()):40===this.peekCodePoint(0)?(this.consumeCodePoint(),{type:sA.FUNCTION_TOKEN,value:A}):{type:sA.IDENT_TOKEN,value:A}},yA.prototype.consumeUrlToken=function(){var A=[];if(this.consumeWhiteSpace(),-1===this.peekCodePoint(0))return{type:sA.URL_TOKEN,value:&quot;&quot;};var e,t=this.peekCodePoint(0);if(39===t||34===t){var r=this.consumeStringToken(this.consumeCodePoint());return r.type===sA.STRING_TOKEN&&(this.consumeWhiteSpace(),-1===this.peekCodePoint(0)||41===this.peekCodePoint(0))?(this.consumeCodePoint(),{type:sA.URL_TOKEN,value:r.value}):(this.consumeBadUrlRemnants(),IA)}for(;;){var B=this.consumeCodePoint();if(-1===B||41===B)return{type:sA.URL_TOKEN,value:l.apply(void 0,A)};if(cA(B))return this.consumeWhiteSpace(),-1===this.peekCodePoint(0)||41===this.peekCodePoint(0)?(this.consumeCodePoint(),{type:sA.URL_TOKEN,value:l.apply(void 0,A)}):(this.consumeBadUrlRemnants(),IA);if(34===B||39===B||40===B||0<=(e=B)&&e<=8||11===e||14<=e&&e<=31||127===e)return this.consumeBadUrlRemnants(),IA;if(92===B){if(!uA(B,this.peekCodePoint(0)))return this.consumeBadUrlRemnants(),IA;A.push(this.consumeEscapedCodePoint())}else A.push(B)}},yA.prototype.consumeWhiteSpace=function(){for(;cA(this.peekCodePoint(0));)this.consumeCodePoint()},yA.prototype.consumeBadUrlRemnants=function(){for(;;){var A=this.consumeCodePoint();if(41===A||-1===A)return;uA(A,this.peekCodePoint(0))&&this.consumeEscapedCodePoint()}},yA.prototype.consumeStringSlice=function(A){for(var e=&quot;&quot;;0<A;){var t=Math.min(6e4,A);e+=l.apply(void 0,this._value.splice(0,t)),A-=t}return this._value.shift(),e},yA.prototype.consumeStringToken=function(A){for(var e=&quot;&quot;,t=0;;){var r=this._value[t];if(-1===r||void 0===r||r===A)return e+=this.consumeStringSlice(t),{type:sA.STRING_TOKEN,value:e};if(10===r)return this._value.splice(0,t),TA;if(92===r){var B=this._value[t+1];-1!==B&&void 0!==B&&(10===B?(e+=this.consumeStringSlice(t),t=-1,this._value.shift()):uA(r,B)&&(e+=this.consumeStringSlice(t),e+=l(this.consumeEscapedCodePoint()),t=-1))}t++}},yA.prototype.consumeNumber=function(){var A=[],e=4,t=this.peekCodePoint(0);for(43!==t&&45!==t||A.push(this.consumeCodePoint());iA(this.peekCodePoint(0));)A.push(this.consumeCodePoint());t=this.peekCodePoint(0);var r=this.peekCodePoint(1);if(46===t&&iA(r))for(A.push(this.consumeCodePoint(),this.consumeCodePoint()),e=8;iA(this.peekCodePoint(0));)A.push(this.consumeCodePoint());t=this.peekCodePoint(0),r=this.peekCodePoint(1);var B=this.peekCodePoint(2);if((69===t||101===t)&&((43===r||45===r)&&iA(B)||iA(r)))for(A.push(this.consumeCodePoint(),this.consumeCodePoint()),e=8;iA(this.peekCodePoint(0));)A.push(this.consumeCodePoint());return[function(A){var e=0,t=1;43!==A[e]&&45!==A[e]||(45===A[e]&&(t=-1),e++);for(var r=[];iA(A[e]);)r.push(A[e++]);var B=r.length?parseInt(l.apply(void 0,r),10):0;46===A[e]&&e++;for(var n=[];iA(A[e]);)n.push(A[e++]);var s=n.length,o=s?parseInt(l.apply(void 0,n),10):0;69!==A[e]&&101!==A[e]||e++;var i=1;43!==A[e]&&45!==A[e]||(45===A[e]&&(i=-1),e++);for(var a=[];iA(A[e]);)a.push(A[e++]);var c=a.length?parseInt(l.apply(void 0,a),10):0;return t*(B+o*Math.pow(10,-s))*Math.pow(10,i*c)}(A),e]},yA.prototype.consumeNumericToken=function(){var A=this.consumeNumber(),e=A[0],t=A[1],r=this.peekCodePoint(0),B=this.peekCodePoint(1),n=this.peekCodePoint(2);if(UA(r,B,n)){var s=this.consumeName();return{type:sA.DIMENSION_TOKEN,number:e,flags:t,unit:s}}return 37===r?(this.consumeCodePoint(),{type:sA.PERCENTAGE_TOKEN,number:e,flags:t}):{type:sA.NUMBER_TOKEN,number:e,flags:t}},yA.prototype.consumeEscapedCodePoint=function(){var A=this.consumeCodePoint();if(aA(A)){for(var e=l(A);aA(this.peekCodePoint(0))&&e.length<6;)e+=l(this.consumeCodePoint());cA(this.peekCodePoint(0))&&this.consumeCodePoint();var t=parseInt(e,16);return 0===t||function(A){return 55296<=A&&A<=57343}(t)||1114111<t?65533:t}return-1===A?65533:A},yA.prototype.consumeName=function(){for(var A=&quot;&quot;;;){var e=this.consumeCodePoint();if(wA(e))A+=l(e);else{if(!uA(e,this.peekCodePoint(0)))return this.reconsumeCodePoint(e),A;A+=l(this.consumeEscapedCodePoint())}}},yA);function yA(){this._value=[]}var _A=(PA.create=function(A){var e=new MA;return e.write(A),new PA(e.read())},PA.parseValue=function(A){return PA.create(A).parseComponentValue()},PA.parseValues=function(A){return PA.create(A).parseComponentValues()},PA.prototype.parseComponentValue=function(){for(var A=this.consumeToken();A.type===sA.WHITESPACE_TOKEN;)A=this.consumeToken();if(A.type===sA.EOF_TOKEN)throw new SyntaxError(&quot;Error parsing CSS component value, unexpected EOF&quot;);this.reconsumeToken(A);for(var e=this.consumeComponentValue();(A=this.consumeToken()).type===sA.WHITESPACE_TOKEN;);if(A.type===sA.EOF_TOKEN)return e;throw new SyntaxError(&quot;Error parsing CSS component value, multiple values found when expecting only one&quot;)},PA.prototype.parseComponentValues=function(){for(var A=[];;){var e=this.consumeComponentValue();if(e.type===sA.EOF_TOKEN)return A;A.push(e),A.push()}},PA.prototype.consumeComponentValue=function(){var A=this.consumeToken();switch(A.type){case sA.LEFT_CURLY_BRACKET_TOKEN:case sA.LEFT_SQUARE_BRACKET_TOKEN:case sA.LEFT_PARENTHESIS_TOKEN:return this.consumeSimpleBlock(A.type);case sA.FUNCTION_TOKEN:return this.consumeFunction(A)}return A},PA.prototype.consumeSimpleBlock=function(A){for(var e={type:A,values:[]},t=this.consumeToken();;){if(t.type===sA.EOF_TOKEN||ne(t,A))return e;this.reconsumeToken(t),e.values.push(this.consumeComponentValue()),t=this.consumeToken()}},PA.prototype.consumeFunction=function(A){for(var e={name:A.value,values:[],type:sA.FUNCTION};;){var t=this.consumeToken();if(t.type===sA.EOF_TOKEN||t.type===sA.RIGHT_PARENTHESIS_TOKEN)return e;this.reconsumeToken(t),e.values.push(this.consumeComponentValue())}},PA.prototype.consumeToken=function(){var A=this._tokens.shift();return void 0===A?bA:A},PA.prototype.reconsumeToken=function(A){this._tokens.unshift(A)},PA);function PA(A){this._tokens=A}function xA(A){return A.type===sA.DIMENSION_TOKEN}function VA(A){return A.type===sA.NUMBER_TOKEN}function zA(A){return A.type===sA.IDENT_TOKEN}function XA(A){return A.type===sA.STRING_TOKEN}function JA(A,e){return zA(A)&&A.value===e}function GA(A){return A.type!==sA.WHITESPACE_TOKEN}function kA(A){return A.type!==sA.WHITESPACE_TOKEN&&A.type!==sA.COMMA_TOKEN}function WA(A){var e=[],t=[];return A.forEach(function(A){if(A.type===sA.COMMA_TOKEN){if(0===t.length)throw new Error(&quot;Error parsing function args, zero tokens for arg&quot;);return e.push(t),void(t=[])}A.type!==sA.WHITESPACE_TOKEN&&t.push(A)}),t.length&&e.push(t),e}function YA(A){return A.type===sA.NUMBER_TOKEN||A.type===sA.DIMENSION_TOKEN}function qA(A){return A.type===sA.PERCENTAGE_TOKEN||YA(A)}function ZA(A){return 1<A.length?[A[0],A[1]]:[A[0]]}function jA(A,e,t){var r=A[0],B=A[1];return[ae(r,e),ae(void 0!==B?B:r,t)]}function $A(A){return A.type===sA.DIMENSION_TOKEN&&(&quot;deg&quot;===A.unit||&quot;grad&quot;===A.unit||&quot;rad&quot;===A.unit||&quot;turn&quot;===A.unit)}function Ae(A){switch(A.filter(zA).map(function(A){return A.value}).join(&quot; &quot;)){case&quot;to bottom right&quot;:case&quot;to right bottom&quot;:case&quot;left top&quot;:case&quot;top left&quot;:return[se,se];case&quot;to top&quot;:case&quot;bottom&quot;:return Qe(0);case&quot;to bottom left&quot;:case&quot;to left bottom&quot;:case&quot;right top&quot;:case&quot;top right&quot;:return[se,ie];case&quot;to right&quot;:case&quot;left&quot;:return Qe(90);case&quot;to top left&quot;:case&quot;to left top&quot;:case&quot;right bottom&quot;:case&quot;bottom right&quot;:return[ie,ie];case&quot;to bottom&quot;:case&quot;top&quot;:return Qe(180);case&quot;to top right&quot;:case&quot;to right top&quot;:case&quot;left bottom&quot;:case&quot;bottom left&quot;:return[ie,se];case&quot;to left&quot;:case&quot;right&quot;:return Qe(270)}return 0}function ee(A){return 0==(255&A)}function te(A){var e=255&A,t=255&A>>8,r=255&A>>16,B=255&A>>24;return e<255?&quot;rgba(&quot;+B+&quot;,&quot;+r+&quot;,&quot;+t+&quot;,&quot;+e/255+&quot;)&quot;:&quot;rgb(&quot;+B+&quot;,&quot;+r+&quot;,&quot;+t+&quot;)&quot;}function re(A,e){if(A.type===sA.NUMBER_TOKEN)return A.number;if(A.type!==sA.PERCENTAGE_TOKEN)return 0;var t=3===e?1:255;return 3===e?A.number/100*t:Math.round(A.number/100*t)}function Be(A){var e=A.filter(kA);if(3===e.length){var t=e.map(re),r=t[0],B=t[1],n=t[2];return ue(r,B,n,1)}if(4!==e.length)return 0;var s=e.map(re),o=(r=s[0],B=s[1],n=s[2],s[3]);return ue(r,B,n,o)}var ne=function(A,e){return e===sA.LEFT_CURLY_BRACKET_TOKEN&&A.type===sA.RIGHT_CURLY_BRACKET_TOKEN||(e===sA.LEFT_SQUARE_BRACKET_TOKEN&&A.type===sA.RIGHT_SQUARE_BRACKET_TOKEN||e===sA.LEFT_PARENTHESIS_TOKEN&&A.type===sA.RIGHT_PARENTHESIS_TOKEN)},se={type:sA.NUMBER_TOKEN,number:0,flags:4},oe={type:sA.PERCENTAGE_TOKEN,number:50,flags:4},ie={type:sA.PERCENTAGE_TOKEN,number:100,flags:4},ae=function(A,e){if(A.type===sA.PERCENTAGE_TOKEN)return A.number/100*e;if(xA(A))switch(A.unit){case&quot;rem&quot;:case&quot;em&quot;:return 16*A.number;case&quot;px&quot;:default:return A.number}return A.number},ce=function(A){if(A.type===sA.DIMENSION_TOKEN)switch(A.unit){case&quot;deg&quot;:return Math.PI*A.number/180;case&quot;grad&quot;:return Math.PI/200*A.number;case&quot;rad&quot;:return A.number;case&quot;turn&quot;:return 2*Math.PI*A.number}throw new Error(&quot;Unsupported angle type&quot;)},Qe=function(A){return Math.PI*A/180},we=function(A){if(A.type===sA.FUNCTION){var e=he[A.name];if(void 0===e)throw new Error('Attempting to parse an unsupported color function &quot;'+A.name+'&quot;');return e(A.values)}if(A.type===sA.HASH_TOKEN){if(3===A.value.length){var t=A.value.substring(0,1),r=A.value.substring(1,2),B=A.value.substring(2,3);return ue(parseInt(t+t,16),parseInt(r+r,16),parseInt(B+B,16),1)}if(4===A.value.length){t=A.value.substring(0,1),r=A.value.substring(1,2),B=A.value.substring(2,3);var n=A.value.substring(3,4);return ue(parseInt(t+t,16),parseInt(r+r,16),parseInt(B+B,16),parseInt(n+n,16)/255)}if(6===A.value.length){t=A.value.substring(0,2),r=A.value.substring(2,4),B=A.value.substring(4,6);return ue(parseInt(t,16),parseInt(r,16),parseInt(B,16),1)}if(8===A.value.length){t=A.value.substring(0,2),r=A.value.substring(2,4),B=A.value.substring(4,6),n=A.value.substring(6,8);return ue(parseInt(t,16),parseInt(r,16),parseInt(B,16),parseInt(n,16)/255)}}if(A.type===sA.IDENT_TOKEN){var s=He[A.value.toUpperCase()];if(void 0!==s)return s}return He.TRANSPARENT},ue=function(A,e,t,r){return(A<<24|e<<16|t<<8|Math.round(255*r)<<0)>>>0};function Ue(A,e,t){return t<0&&(t+=1),1<=t&&(t-=1),t<1/6?(e-A)*t*6+A:t<.5?e:t<2/3?6*(e-A)*(2/3-t)+A:A}function le(A){var e=A.filter(kA),t=e[0],r=e[1],B=e[2],n=e[3],s=(t.type===sA.NUMBER_TOKEN?Qe(t.number):ce(t))/(2*Math.PI),o=qA(r)?r.number/100:0,i=qA(B)?B.number/100:0,a=void 0!==n&&qA(n)?ae(n,1):1;if(0==o)return ue(255*i,255*i,255*i,1);var c=i<=.5?i*(1+o):i+o-i*o,Q=2*i-c,w=Ue(Q,c,s+1/3),u=Ue(Q,c,s),U=Ue(Q,c,s-1/3);return ue(255*w,255*u,255*U,a)}var Ce,ge,Ee,Fe,he={hsl:le,hsla:le,rgb:Be,rgba:Be},He={ALICEBLUE:4042850303,ANTIQUEWHITE:4209760255,AQUA:16777215,AQUAMARINE:2147472639,AZURE:4043309055,BEIGE:4126530815,BISQUE:4293182719,BLACK:255,BLANCHEDALMOND:4293643775,BLUE:65535,BLUEVIOLET:2318131967,BROWN:2771004159,BURLYWOOD:3736635391,CADETBLUE:1604231423,CHARTREUSE:2147418367,CHOCOLATE:3530104575,CORAL:4286533887,CORNFLOWERBLUE:1687547391,CORNSILK:4294499583,CRIMSON:3692313855,CYAN:16777215,DARKBLUE:35839,DARKCYAN:9145343,DARKGOLDENROD:3095837695,DARKGRAY:2846468607,DARKGREEN:6553855,DARKGREY:2846468607,DARKKHAKI:3182914559,DARKMAGENTA:2332068863,DARKOLIVEGREEN:1433087999,DARKORANGE:4287365375,DARKORCHID:2570243327,DARKRED:2332033279,DARKSALMON:3918953215,DARKSEAGREEN:2411499519,DARKSLATEBLUE:1211993087,DARKSLATEGRAY:793726975,DARKSLATEGREY:793726975,DARKTURQUOISE:13554175,DARKVIOLET:2483082239,DEEPPINK:4279538687,DEEPSKYBLUE:12582911,DIMGRAY:1768516095,DIMGREY:1768516095,DODGERBLUE:512819199,FIREBRICK:2988581631,FLORALWHITE:4294635775,FORESTGREEN:579543807,FUCHSIA:4278255615,GAINSBORO:3705462015,GHOSTWHITE:4177068031,GOLD:4292280575,GOLDENROD:3668254975,GRAY:2155905279,GREEN:8388863,GREENYELLOW:2919182335,GREY:2155905279,HONEYDEW:4043305215,HOTPINK:4285117695,INDIANRED:3445382399,INDIGO:1258324735,IVORY:4294963455,KHAKI:4041641215,LAVENDER:3873897215,LAVENDERBLUSH:4293981695,LAWNGREEN:2096890111,LEMONCHIFFON:4294626815,LIGHTBLUE:2916673279,LIGHTCORAL:4034953471,LIGHTCYAN:3774873599,LIGHTGOLDENRODYELLOW:4210742015,LIGHTGRAY:3553874943,LIGHTGREEN:2431553791,LIGHTGREY:3553874943,LIGHTPINK:4290167295,LIGHTSALMON:4288707327,LIGHTSEAGREEN:548580095,LIGHTSKYBLUE:2278488831,LIGHTSLATEGRAY:2005441023,LIGHTSLATEGREY:2005441023,LIGHTSTEELBLUE:2965692159,LIGHTYELLOW:4294959359,LIME:16711935,LIMEGREEN:852308735,LINEN:4210091775,MAGENTA:4278255615,MAROON:2147483903,MEDIUMAQUAMARINE:1724754687,MEDIUMBLUE:52735,MEDIUMORCHID:3126187007,MEDIUMPURPLE:2473647103,MEDIUMSEAGREEN:1018393087,MEDIUMSLATEBLUE:2070474495,MEDIUMSPRINGGREEN:16423679,MEDIUMTURQUOISE:1221709055,MEDIUMVIOLETRED:3340076543,MIDNIGHTBLUE:421097727,MINTCREAM:4127193855,MISTYROSE:4293190143,MOCCASIN:4293178879,NAVAJOWHITE:4292783615,NAVY:33023,OLDLACE:4260751103,OLIVE:2155872511,OLIVEDRAB:1804477439,ORANGE:4289003775,ORANGERED:4282712319,ORCHID:3664828159,PALEGOLDENROD:4008225535,PALEGREEN:2566625535,PALETURQUOISE:2951671551,PALEVIOLETRED:3681588223,PAPAYAWHIP:4293907967,PEACHPUFF:4292524543,PERU:3448061951,PINK:4290825215,PLUM:3718307327,POWDERBLUE:2967529215,PURPLE:2147516671,REBECCAPURPLE:1714657791,RED:4278190335,ROSYBROWN:3163525119,ROYALBLUE:1097458175,SADDLEBROWN:2336560127,SALMON:4202722047,SANDYBROWN:4104413439,SEAGREEN:780883967,SEASHELL:4294307583,SIENNA:2689740287,SILVER:3233857791,SKYBLUE:2278484991,SLATEBLUE:1784335871,SLATEGRAY:1887473919,SLATEGREY:1887473919,SNOW:4294638335,SPRINGGREEN:16744447,STEELBLUE:1182971135,TAN:3535047935,TEAL:8421631,THISTLE:3636451583,TOMATO:4284696575,TRANSPARENT:0,TURQUOISE:1088475391,VIOLET:4001558271,WHEAT:4125012991,WHITE:4294967295,WHITESMOKE:4126537215,YELLOW:4294902015,YELLOWGREEN:2597139199};(ge=Ce||(Ce={}))[ge.VALUE=0]=&quot;VALUE&quot;,ge[ge.LIST=1]=&quot;LIST&quot;,ge[ge.IDENT_VALUE=2]=&quot;IDENT_VALUE&quot;,ge[ge.TYPE_VALUE=3]=&quot;TYPE_VALUE&quot;,ge[ge.TOKEN_VALUE=4]=&quot;TOKEN_VALUE&quot;,(Fe=Ee||(Ee={}))[Fe.BORDER_BOX=0]=&quot;BORDER_BOX&quot;,Fe[Fe.PADDING_BOX=1]=&quot;PADDING_BOX&quot;;function de(A){var e=we(A[0]),t=A[1];return t&&qA(t)?{color:e,stop:t}:{color:e,stop:null}}function fe(A,t){var e=A[0],r=A[A.length-1];null===e.stop&&(e.stop=se),null===r.stop&&(r.stop=ie);for(var B=[],n=0,s=0;s<A.length;s++){var o=A[s].stop;if(null!==o){var i=ae(o,t);n<i?B.push(i):B.push(n),n=i}else B.push(null)}var a=null;for(s=0;s<B.length;s++){var c=B[s];if(null===c)null===a&&(a=s);else if(null!==a){for(var Q=s-a,w=(c-B[a-1])/(1+Q),u=1;u<=Q;u++)B[a+u-1]=w*u;a=null}}return A.map(function(A,e){return{color:A.color,stop:Math.max(Math.min(1,B[e]/t),0)}})}function pe(A,e,t){var r=&quot;number&quot;==typeof A?A:function(A,e,t){var r=e/2,B=t/2,n=ae(A[0],e)-r,s=B-ae(A[1],t);return(Math.atan2(s,n)+2*Math.PI)%(2*Math.PI)}(A,e,t),B=Math.abs(e*Math.sin(r))+Math.abs(t*Math.cos(r)),n=e/2,s=t/2,o=B/2,i=Math.sin(r-Math.PI/2)*o,a=Math.cos(r-Math.PI/2)*o;return[B,n-a,n+a,s-i,s+i]}function Ne(A,e){return Math.sqrt(A*A+e*e)}function Ke(A,e,n,s,o){return[[0,0],[0,e],[A,0],[A,e]].reduce(function(A,e){var t=e[0],r=e[1],B=Ne(n-t,s-r);return(o?B<A.optimumDistance:B>A.optimumDistance)?{optimumCorner:e,optimumDistance:B}:A},{optimumDistance:o?1/0:-1/0,optimumCorner:null}).optimumCorner}function Ie(A){var B=Qe(180),n=[];return WA(A).forEach(function(A,e){if(0===e){var t=A[0];if(t.type===sA.IDENT_TOKEN&&-1!==[&quot;top&quot;,&quot;left&quot;,&quot;right&quot;,&quot;bottom&quot;].indexOf(t.value))return void(B=Ae(A));if($A(t))return void(B=(ce(t)+Qe(270))%Qe(360))}var r=de(A);n.push(r)}),{angle:B,stops:n,type:xe.LINEAR_GRADIENT}}function Te(A){return 0===A[0]&&255===A[1]&&0===A[2]&&255===A[3]}var me={name:&quot;background-clip&quot;,initialValue:&quot;border-box&quot;,prefix:!(Fe[Fe.CONTENT_BOX=2]=&quot;CONTENT_BOX&quot;),type:Ce.LIST,parse:function(A){return A.map(function(A){if(zA(A))switch(A.value){case&quot;padding-box&quot;:return Ee.PADDING_BOX;case&quot;content-box&quot;:return Ee.CONTENT_BOX}return Ee.BORDER_BOX})}},Re={name:&quot;background-color&quot;,initialValue:&quot;transparent&quot;,prefix:!1,type:Ce.TYPE_VALUE,format:&quot;color&quot;},Le=function(A,e,t,r,B){var n=&quot;http://www.w3.org/2000/svg&quot;,s=document.createElementNS(n,&quot;svg&quot;),o=document.createElementNS(n,&quot;foreignObject&quot;);return s.setAttributeNS(null,&quot;width&quot;,A.toString()),s.setAttributeNS(null,&quot;height&quot;,e.toString()),o.setAttributeNS(null,&quot;width&quot;,&quot;100%&quot;),o.setAttributeNS(null,&quot;height&quot;,&quot;100%&quot;),o.setAttributeNS(null,&quot;x&quot;,t.toString()),o.setAttributeNS(null,&quot;y&quot;,r.toString()),o.setAttributeNS(null,&quot;externalResourcesRequired&quot;,&quot;true&quot;),s.appendChild(o),o.appendChild(B),s},Oe=function(r){return new Promise(function(A,e){var t=new Image;t.onload=function(){return A(t)},t.onerror=e,t.src=&quot;data:image/svg+xml;charset=utf-8,&quot;+encodeURIComponent((new XMLSerializer).serializeToString(r))})},ve={get SUPPORT_RANGE_BOUNDS(){var A=function(A){if(A.createRange){var e=A.createRange();if(e.getBoundingClientRect){var t=A.createElement(&quot;boundtest&quot;);t.style.height=&quot;123px&quot;,t.style.display=&quot;block&quot;,A.body.appendChild(t),e.selectNode(t);var r=e.getBoundingClientRect(),B=Math.round(r.height);if(A.body.removeChild(t),123===B)return!0}}return!1}(document);return Object.defineProperty(ve,&quot;SUPPORT_RANGE_BOUNDS&quot;,{value:A}),A},get SUPPORT_SVG_DRAWING(){var A=function(A){var e=new Image,t=A.createElement(&quot;canvas&quot;),r=t.getContext(&quot;2d&quot;);if(!r)return!1;e.src=&quot;data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'></svg>&quot;;try{r.drawImage(e,0,0),t.toDataURL()}catch(A){return!1}return!0}(document);return Object.defineProperty(ve,&quot;SUPPORT_SVG_DRAWING&quot;,{value:A}),A},get SUPPORT_FOREIGNOBJECT_DRAWING(){var A=&quot;function&quot;==typeof Array.from&&&quot;function&quot;==typeof window.fetch?function(r){var A=r.createElement(&quot;canvas&quot;),B=100;A.width=B,A.height=B;var n=A.getContext(&quot;2d&quot;);if(!n)return Promise.reject(!1);n.fillStyle=&quot;rgb(0, 255, 0)&quot;,n.fillRect(0,0,B,B);var e=new Image,s=A.toDataURL();e.src=s;var t=Le(B,B,0,0,e);return n.fillStyle=&quot;red&quot;,n.fillRect(0,0,B,B),Oe(t).then(function(A){n.drawImage(A,0,0);var e=n.getImageData(0,0,B,B).data;n.fillStyle=&quot;red&quot;,n.fillRect(0,0,B,B);var t=r.createElement(&quot;div&quot;);return t.style.backgroundImage=&quot;url(&quot;+s+&quot;)&quot;,t.style.height=&quot;100px&quot;,Te(e)?Oe(Le(B,B,0,0,t)):Promise.reject(!1)}).then(function(A){return n.drawImage(A,0,0),Te(n.getImageData(0,0,B,B).data)}).catch(function(){return!1})}(document):Promise.resolve(!1);return Object.defineProperty(ve,&quot;SUPPORT_FOREIGNOBJECT_DRAWING&quot;,{value:A}),A},get SUPPORT_CORS_IMAGES(){var A=void 0!==(new Image).crossOrigin;return Object.defineProperty(ve,&quot;SUPPORT_CORS_IMAGES&quot;,{value:A}),A},get SUPPORT_RESPONSE_TYPE(){var A=&quot;string&quot;==typeof(new XMLHttpRequest).responseType;return Object.defineProperty(ve,&quot;SUPPORT_RESPONSE_TYPE&quot;,{value:A}),A},get SUPPORT_CORS_XHR(){var A=&quot;withCredentials&quot;in new XMLHttpRequest;return Object.defineProperty(ve,&quot;SUPPORT_CORS_XHR&quot;,{value:A}),A}},De=(Se.prototype.debug=function(){for(var A=[],e=0;e<arguments.length;e++)A[e]=arguments[e];&quot;undefined&quot;!=typeof window&&window.console&&&quot;function&quot;==typeof console.debug?console.debug.apply(console,[this.id,this.getTime()+&quot;ms&quot;].concat(A)):this.info.apply(this,A)},Se.prototype.getTime=function(){return Date.now()-this.start},Se.create=function(A){Se.instances[A]=new Se(A)},Se.destroy=function(A){delete Se.instances[A]},Se.getInstance=function(A){var e=Se.instances[A];if(void 0===e)throw new Error(&quot;No logger instance found with id &quot;+A);return e},Se.prototype.info=function(){for(var A=[],e=0;e<arguments.length;e++)A[e]=arguments[e];&quot;undefined&quot;!=typeof window&&window.console&&&quot;function&quot;==typeof console.info&&console.info.apply(console,[this.id,this.getTime()+&quot;ms&quot;].concat(A))},Se.prototype.error=function(){for(var A=[],e=0;e<arguments.length;e++)A[e]=arguments[e];&quot;undefined&quot;!=typeof window&&window.console&&&quot;function&quot;==typeof console.error?console.error.apply(console,[this.id,this.getTime()+&quot;ms&quot;].concat(A)):this.info.apply(this,A)},Se.instances={},Se);function Se(A){this.id=A,this.start=Date.now()}var be=(Me.create=function(A,e){return Me._caches[A]=new ye(A,e)},Me.destroy=function(A){delete Me._caches[A]},Me.open=function(A){var e=Me._caches[A];if(void 0!==e)return e;throw new Error('Cache with key &quot;'+A+'&quot; not found')},Me.getOrigin=function(A){var e=Me._link;return e?(e.href=A,e.href=e.href,e.protocol+e.hostname+e.port):&quot;about:blank&quot;},Me.isSameOrigin=function(A){return Me.getOrigin(A)===Me._origin},Me.setContext=function(A){Me._link=A.document.createElement(&quot;a&quot;),Me._origin=Me.getOrigin(A.location.href)},Me.getInstance=function(){var A=Me._current;if(null===A)throw new Error(&quot;No cache instance attached&quot;);return A},Me.attachInstance=function(A){Me._current=A},Me.detachInstance=function(){Me._current=null},Me._caches={},Me._origin=&quot;about:blank&quot;,Me._current=null,Me);function Me(){}var ye=(_e.prototype.addImage=function(A){var e=Promise.resolve();return this.has(A)||(Ye(A)||Ge(A))&&(this._cache[A]=this.loadImage(A)),e},_e.prototype.match=function(A){return this._cache[A]},_e.prototype.loadImage=function(s){return B(this,void 0,void 0,function(){var e,r,t,B,n=this;return b(this,function(A){switch(A.label){case 0:return e=be.isSameOrigin(s),r=!ke(s)&&!0===this._options.useCORS&&ve.SUPPORT_CORS_IMAGES&&!e,t=!ke(s)&&!e&&&quot;string&quot;==typeof this._options.proxy&&ve.SUPPORT_CORS_XHR&&!r,e||!1!==this._options.allowTaint||ke(s)||t||r?(B=s,t?[4,this.proxy(B)]:[3,2]):[2];case 1:B=A.sent(),A.label=2;case 2:return De.getInstance(this.id).debug(&quot;Added image &quot;+s.substring(0,256)),[4,new Promise(function(A,e){var t=new Image;t.onload=function(){return A(t)},t.onerror=e,(We(B)||r)&&(t.crossOrigin=&quot;anonymous&quot;),t.src=B,!0===t.complete&&setTimeout(function(){return A(t)},500),0<n._options.imageTimeout&&setTimeout(function(){return e(&quot;Timed out (&quot;+n._options.imageTimeout+&quot;ms) loading image&quot;)},n._options.imageTimeout)})];case 3:return[2,A.sent()]}})})},_e.prototype.has=function(A){return void 0!==this._cache[A]},_e.prototype.keys=function(){return Promise.resolve(Object.keys(this._cache))},_e.prototype.proxy=function(n){var s=this,o=this._options.proxy;if(!o)throw new Error(&quot;No proxy defined&quot;);var i=n.substring(0,256);return new Promise(function(e,t){var r=ve.SUPPORT_RESPONSE_TYPE?&quot;blob&quot;:&quot;text&quot;,B=new XMLHttpRequest;if(B.onload=function(){if(200===B.status)if(&quot;text&quot;==r)e(B.response);else{var A=new FileReader;A.addEventListener(&quot;load&quot;,function(){return e(A.result)},!1),A.addEventListener(&quot;error&quot;,function(A){return t(A)},!1),A.readAsDataURL(B.response)}else t(&quot;Failed to proxy resource &quot;+i+&quot; with status code &quot;+B.status)},B.onerror=t,B.open(&quot;GET&quot;,o+&quot;?url=&quot;+encodeURIComponent(n)+&quot;&responseType=&quot;+r),&quot;text&quot;!=r&&B instanceof XMLHttpRequest&&(B.responseType=r),s._options.imageTimeout){var A=s._options.imageTimeout;B.timeout=A,B.ontimeout=function(){return t(&quot;Timed out (&quot;+A+&quot;ms) proxying &quot;+i)}}B.send()})},_e);function _e(A,e){this.id=A,this._options=e,this._cache={}}function Pe(A){var B=rt.CIRCLE,n=nt.FARTHEST_CORNER,s=[],o=[];return WA(A).forEach(function(A,e){var t=!0;if(0===e?t=A.reduce(function(A,e){if(zA(e))switch(e.value){case&quot;center&quot;:return o.push(oe),!1;case&quot;top&quot;:case&quot;left&quot;:return o.push(se),!1;case&quot;right&quot;:case&quot;bottom&quot;:return o.push(ie),!1}else if(qA(e)||YA(e))return o.push(e),!1;return A},t):1===e&&(t=A.reduce(function(A,e){if(zA(e))switch(e.value){case&quot;circle&quot;:return B=rt.CIRCLE,!1;case et:return B=rt.ELLIPSE,!1;case tt:case Ze:return n=nt.CLOSEST_SIDE,!1;case je:return n=nt.FARTHEST_SIDE,!1;case $e:return n=nt.CLOSEST_CORNER,!1;case&quot;cover&quot;:case At:return n=nt.FARTHEST_CORNER,!1}else if(YA(e)||qA(e))return Array.isArray(n)||(n=[]),n.push(e),!1;return A},t)),t){var r=de(A);s.push(r)}}),{size:n,shape:B,stops:s,position:o,type:xe.RADIAL_GRADIENT}}var xe,Ve,ze=/^data:image\/svg\+xml/i,Xe=/^data:image\/.*;base64,/i,Je=/^data:image\/.*/i,Ge=function(A){return ve.SUPPORT_SVG_DRAWING||!qe(A)},ke=function(A){return Je.test(A)},We=function(A){return Xe.test(A)},Ye=function(A){return&quot;blob&quot;===A.substr(0,4)},qe=function(A){return&quot;svg&quot;===A.substr(-3).toLowerCase()||ze.test(A)},Ze=&quot;closest-side&quot;,je=&quot;farthest-side&quot;,$e=&quot;closest-corner&quot;,At=&quot;farthest-corner&quot;,et=&quot;ellipse&quot;,tt=&quot;contain&quot;;(Ve=xe||(xe={}))[Ve.URL=0]=&quot;URL&quot;,Ve[Ve.LINEAR_GRADIENT=1]=&quot;LINEAR_GRADIENT&quot;,Ve[Ve.RADIAL_GRADIENT=2]=&quot;RADIAL_GRADIENT&quot;;var rt,Bt,nt,st;(Bt=rt||(rt={}))[Bt.CIRCLE=0]=&quot;CIRCLE&quot;,Bt[Bt.ELLIPSE=1]=&quot;ELLIPSE&quot;,(st=nt||(nt={}))[st.CLOSEST_SIDE=0]=&quot;CLOSEST_SIDE&quot;,st[st.FARTHEST_SIDE=1]=&quot;FARTHEST_SIDE&quot;,st[st.CLOSEST_CORNER=2]=&quot;CLOSEST_CORNER&quot;,st[st.FARTHEST_CORNER=3]=&quot;FARTHEST_CORNER&quot;;var ot,it,at=function(A){if(A.type===sA.URL_TOKEN){var e={url:A.value,type:xe.URL};return be.getInstance().addImage(A.value),e}if(A.type!==sA.FUNCTION)throw new Error(&quot;Unsupported image type&quot;);var t=ct[A.name];if(void 0===t)throw new Error('Attempting to parse an unsupported image function &quot;'+A.name+'&quot;');return t(A.values)},ct={&quot;linear-gradient&quot;:function(A){var B=Qe(180),n=[];return WA(A).forEach(function(A,e){if(0===e){var t=A[0];if(t.type===sA.IDENT_TOKEN&&&quot;to&quot;===t.value)return void(B=Ae(A));if($A(t))return void(B=ce(t))}var r=de(A);n.push(r)}),{angle:B,stops:n,type:xe.LINEAR_GRADIENT}},&quot;-moz-linear-gradient&quot;:Ie,&quot;-ms-linear-gradient&quot;:Ie,&quot;-o-linear-gradient&quot;:Ie,&quot;-webkit-linear-gradient&quot;:Ie,&quot;radial-gradient&quot;:function(A){var n=rt.CIRCLE,s=nt.FARTHEST_CORNER,o=[],i=[];return WA(A).forEach(function(A,e){var t=!0;if(0===e){var r=!1;t=A.reduce(function(A,e){if(r)if(zA(e))switch(e.value){case&quot;center&quot;:return i.push(oe),A;case&quot;top&quot;:case&quot;left&quot;:return i.push(se),A;case&quot;right&quot;:case&quot;bottom&quot;:return i.push(ie),A}else(qA(e)||YA(e))&&i.push(e);else if(zA(e))switch(e.value){case&quot;circle&quot;:return n=rt.CIRCLE,!1;case et:return n=rt.ELLIPSE,!1;case&quot;at&quot;:return!(r=!0);case Ze:return s=nt.CLOSEST_SIDE,!1;case&quot;cover&quot;:case je:return s=nt.FARTHEST_SIDE,!1;case tt:case $e:return s=nt.CLOSEST_CORNER,!1;case At:return s=nt.FARTHEST_CORNER,!1}else if(YA(e)||qA(e))return Array.isArray(s)||(s=[]),s.push(e),!1;return A},t)}if(t){var B=de(A);o.push(B)}}),{size:s,shape:n,stops:o,position:i,type:xe.RADIAL_GRADIENT}},&quot;-moz-radial-gradient&quot;:Pe,&quot;-ms-radial-gradient&quot;:Pe,&quot;-o-radial-gradient&quot;:Pe,&quot;-webkit-radial-gradient&quot;:Pe,&quot;-webkit-gradient&quot;:function(A){var e=Qe(180),s=[],o=xe.LINEAR_GRADIENT,t=rt.CIRCLE,r=nt.FARTHEST_CORNER;return WA(A).forEach(function(A,e){var t=A[0];if(0===e){if(zA(t)&&&quot;linear&quot;===t.value)return void(o=xe.LINEAR_GRADIENT);if(zA(t)&&&quot;radial&quot;===t.value)return void(o=xe.RADIAL_GRADIENT)}if(t.type===sA.FUNCTION)if(&quot;from&quot;===t.name){var r=we(t.values[0]);s.push({stop:se,color:r})}else if(&quot;to&quot;===t.name)r=we(t.values[0]),s.push({stop:ie,color:r});else if(&quot;color-stop&quot;===t.name){var B=t.values.filter(kA);if(2===B.length){r=we(B[1]);var n=B[0];VA(n)&&s.push({stop:{type:sA.PERCENTAGE_TOKEN,number:100*n.number,flags:n.flags},color:r})}}}),o===xe.LINEAR_GRADIENT?{angle:(e+Qe(180))%Qe(360),stops:s,type:o}:{size:r,shape:t,stops:s,position:[],type:o}}},Qt={name:&quot;background-image&quot;,initialValue:&quot;none&quot;,type:Ce.LIST,prefix:!1,parse:function(A){if(0===A.length)return[];var e=A[0];return e.type===sA.IDENT_TOKEN&&&quot;none&quot;===e.value?[]:A.filter(kA).map(at)}},wt={name:&quot;background-origin&quot;,initialValue:&quot;border-box&quot;,prefix:!1,type:Ce.LIST,parse:function(A){return A.map(function(A){if(zA(A))switch(A.value){case&quot;padding-box&quot;:return 1;case&quot;content-box&quot;:return 2}return 0})}},ut={name:&quot;background-position&quot;,initialValue:&quot;0% 0%&quot;,type:Ce.LIST,prefix:!1,parse:function(A){return WA(A).map(function(A){return A.filter(qA)}).map(ZA)}};(it=ot||(ot={}))[it.REPEAT=0]=&quot;REPEAT&quot;,it[it.NO_REPEAT=1]=&quot;NO_REPEAT&quot;,it[it.REPEAT_X=2]=&quot;REPEAT_X&quot;;var Ut,lt,Ct={name:&quot;background-repeat&quot;,initialValue:&quot;repeat&quot;,prefix:!(it[it.REPEAT_Y=3]=&quot;REPEAT_Y&quot;),type:Ce.LIST,parse:function(A){return WA(A).map(function(A){return A.filter(zA).map(function(A){return A.value}).join(&quot; &quot;)}).map(gt)}},gt=function(A){switch(A){case&quot;no-repeat&quot;:return ot.NO_REPEAT;case&quot;repeat-x&quot;:case&quot;repeat no-repeat&quot;:return ot.REPEAT_X;case&quot;repeat-y&quot;:case&quot;no-repeat repeat&quot;:return ot.REPEAT_Y;case&quot;repeat&quot;:default:return ot.REPEAT}};(lt=Ut||(Ut={})).AUTO=&quot;auto&quot;,lt.CONTAIN=&quot;contain&quot;;function Et(A){return{name:&quot;border-&quot;+A+&quot;-color&quot;,initialValue:&quot;transparent&quot;,prefix:!1,type:Ce.TYPE_VALUE,format:&quot;color&quot;}}function Ft(A){return{name:&quot;border-radius-&quot;+A,initialValue:&quot;0 0&quot;,prefix:!1,type:Ce.LIST,parse:function(A){return ZA(A.filter(qA))}}}var ht,Ht,dt={name:&quot;background-size&quot;,initialValue:&quot;0&quot;,prefix:!(lt.COVER=&quot;cover&quot;),type:Ce.LIST,parse:function(A){return WA(A).map(function(A){return A.filter(ft)})}},ft=function(A){return zA(A)||qA(A)},pt=Et(&quot;top&quot;),Nt=Et(&quot;right&quot;),Kt=Et(&quot;bottom&quot;),It=Et(&quot;left&quot;),Tt=Ft(&quot;top-left&quot;),mt=Ft(&quot;top-right&quot;),Rt=Ft(&quot;bottom-right&quot;),Lt=Ft(&quot;bottom-left&quot;);(Ht=ht||(ht={}))[Ht.NONE=0]=&quot;NONE&quot;,Ht[Ht.SOLID=1]=&quot;SOLID&quot;;function Ot(A){return{name:&quot;border-&quot;+A+&quot;-style&quot;,initialValue:&quot;solid&quot;,prefix:!1,type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;none&quot;:return ht.NONE}return ht.SOLID}}}function vt(A){return{name:&quot;border-&quot;+A+&quot;-width&quot;,initialValue:&quot;0&quot;,type:Ce.VALUE,prefix:!1,parse:function(A){return xA(A)?A.number:0}}}var Dt,St,bt=Ot(&quot;top&quot;),Mt=Ot(&quot;right&quot;),yt=Ot(&quot;bottom&quot;),_t=Ot(&quot;left&quot;),Pt=vt(&quot;top&quot;),xt=vt(&quot;right&quot;),Vt=vt(&quot;bottom&quot;),zt=vt(&quot;left&quot;),Xt={name:&quot;color&quot;,initialValue:&quot;transparent&quot;,prefix:!1,type:Ce.TYPE_VALUE,format:&quot;color&quot;},Jt={name:&quot;display&quot;,initialValue:&quot;inline-block&quot;,prefix:!1,type:Ce.LIST,parse:function(A){return A.filter(zA).reduce(function(A,e){return A|Gt(e.value)},0)}},Gt=function(A){switch(A){case&quot;block&quot;:return 2;case&quot;inline&quot;:return 4;case&quot;run-in&quot;:return 8;case&quot;flow&quot;:return 16;case&quot;flow-root&quot;:return 32;case&quot;table&quot;:return 64;case&quot;flex&quot;:case&quot;-webkit-flex&quot;:return 128;case&quot;grid&quot;:return 256;case&quot;ruby&quot;:return 512;case&quot;subgrid&quot;:return 1024;case&quot;list-item&quot;:return 2048;case&quot;table-row-group&quot;:return 4096;case&quot;table-header-group&quot;:return 8192;case&quot;table-footer-group&quot;:return 16384;case&quot;table-row&quot;:return 32768;case&quot;table-cell&quot;:return 65536;case&quot;table-column-group&quot;:return 131072;case&quot;table-column&quot;:return 262144;case&quot;table-caption&quot;:return 524288;case&quot;ruby-base&quot;:return 1048576;case&quot;ruby-text&quot;:return 2097152;case&quot;ruby-base-container&quot;:return 4194304;case&quot;ruby-text-container&quot;:return 8388608;case&quot;contents&quot;:return 16777216;case&quot;inline-block&quot;:return 33554432;case&quot;inline-list-item&quot;:return 67108864;case&quot;inline-table&quot;:return 134217728;case&quot;inline-flex&quot;:return 268435456;case&quot;inline-grid&quot;:return 536870912}return 0};(St=Dt||(Dt={}))[St.NONE=0]=&quot;NONE&quot;,St[St.LEFT=1]=&quot;LEFT&quot;,St[St.RIGHT=2]=&quot;RIGHT&quot;,St[St.INLINE_START=3]=&quot;INLINE_START&quot;;var kt,Wt,Yt,qt,Zt={name:&quot;float&quot;,initialValue:&quot;none&quot;,prefix:!(St[St.INLINE_END=4]=&quot;INLINE_END&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;left&quot;:return Dt.LEFT;case&quot;right&quot;:return Dt.RIGHT;case&quot;inline-start&quot;:return Dt.INLINE_START;case&quot;inline-end&quot;:return Dt.INLINE_END}return Dt.NONE}},jt={name:&quot;letter-spacing&quot;,initialValue:&quot;0&quot;,prefix:!1,type:Ce.VALUE,parse:function(A){return A.type===sA.IDENT_TOKEN&&&quot;normal&quot;===A.value?0:A.type===sA.NUMBER_TOKEN?A.number:A.type===sA.DIMENSION_TOKEN?A.number:0}},$t={name:&quot;line-break&quot;,initialValue:(Wt=kt||(kt={})).NORMAL=&quot;normal&quot;,prefix:!(Wt.STRICT=&quot;strict&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;strict&quot;:return kt.STRICT;case&quot;normal&quot;:default:return kt.NORMAL}}},Ar={name:&quot;line-height&quot;,initialValue:&quot;normal&quot;,prefix:!1,type:Ce.TOKEN_VALUE},er={name:&quot;list-style-image&quot;,initialValue:&quot;none&quot;,type:Ce.VALUE,prefix:!1,parse:function(A){return A.type===sA.IDENT_TOKEN&&&quot;none&quot;===A.value?null:at(A)}};(qt=Yt||(Yt={}))[qt.INSIDE=0]=&quot;INSIDE&quot;;var tr,rr,Br={name:&quot;list-style-position&quot;,initialValue:&quot;outside&quot;,prefix:!(qt[qt.OUTSIDE=1]=&quot;OUTSIDE&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;inside&quot;:return Yt.INSIDE;case&quot;outside&quot;:default:return Yt.OUTSIDE}}};(rr=tr||(tr={}))[rr.NONE=-1]=&quot;NONE&quot;,rr[rr.DISC=0]=&quot;DISC&quot;,rr[rr.CIRCLE=1]=&quot;CIRCLE&quot;,rr[rr.SQUARE=2]=&quot;SQUARE&quot;,rr[rr.DECIMAL=3]=&quot;DECIMAL&quot;,rr[rr.CJK_DECIMAL=4]=&quot;CJK_DECIMAL&quot;,rr[rr.DECIMAL_LEADING_ZERO=5]=&quot;DECIMAL_LEADING_ZERO&quot;,rr[rr.LOWER_ROMAN=6]=&quot;LOWER_ROMAN&quot;,rr[rr.UPPER_ROMAN=7]=&quot;UPPER_ROMAN&quot;,rr[rr.LOWER_GREEK=8]=&quot;LOWER_GREEK&quot;,rr[rr.LOWER_ALPHA=9]=&quot;LOWER_ALPHA&quot;,rr[rr.UPPER_ALPHA=10]=&quot;UPPER_ALPHA&quot;,rr[rr.ARABIC_INDIC=11]=&quot;ARABIC_INDIC&quot;,rr[rr.ARMENIAN=12]=&quot;ARMENIAN&quot;,rr[rr.BENGALI=13]=&quot;BENGALI&quot;,rr[rr.CAMBODIAN=14]=&quot;CAMBODIAN&quot;,rr[rr.CJK_EARTHLY_BRANCH=15]=&quot;CJK_EARTHLY_BRANCH&quot;,rr[rr.CJK_HEAVENLY_STEM=16]=&quot;CJK_HEAVENLY_STEM&quot;,rr[rr.CJK_IDEOGRAPHIC=17]=&quot;CJK_IDEOGRAPHIC&quot;,rr[rr.DEVANAGARI=18]=&quot;DEVANAGARI&quot;,rr[rr.ETHIOPIC_NUMERIC=19]=&quot;ETHIOPIC_NUMERIC&quot;,rr[rr.GEORGIAN=20]=&quot;GEORGIAN&quot;,rr[rr.GUJARATI=21]=&quot;GUJARATI&quot;,rr[rr.GURMUKHI=22]=&quot;GURMUKHI&quot;,rr[rr.HEBREW=22]=&quot;HEBREW&quot;,rr[rr.HIRAGANA=23]=&quot;HIRAGANA&quot;,rr[rr.HIRAGANA_IROHA=24]=&quot;HIRAGANA_IROHA&quot;,rr[rr.JAPANESE_FORMAL=25]=&quot;JAPANESE_FORMAL&quot;,rr[rr.JAPANESE_INFORMAL=26]=&quot;JAPANESE_INFORMAL&quot;,rr[rr.KANNADA=27]=&quot;KANNADA&quot;,rr[rr.KATAKANA=28]=&quot;KATAKANA&quot;,rr[rr.KATAKANA_IROHA=29]=&quot;KATAKANA_IROHA&quot;,rr[rr.KHMER=30]=&quot;KHMER&quot;,rr[rr.KOREAN_HANGUL_FORMAL=31]=&quot;KOREAN_HANGUL_FORMAL&quot;,rr[rr.KOREAN_HANJA_FORMAL=32]=&quot;KOREAN_HANJA_FORMAL&quot;,rr[rr.KOREAN_HANJA_INFORMAL=33]=&quot;KOREAN_HANJA_INFORMAL&quot;,rr[rr.LAO=34]=&quot;LAO&quot;,rr[rr.LOWER_ARMENIAN=35]=&quot;LOWER_ARMENIAN&quot;,rr[rr.MALAYALAM=36]=&quot;MALAYALAM&quot;,rr[rr.MONGOLIAN=37]=&quot;MONGOLIAN&quot;,rr[rr.MYANMAR=38]=&quot;MYANMAR&quot;,rr[rr.ORIYA=39]=&quot;ORIYA&quot;,rr[rr.PERSIAN=40]=&quot;PERSIAN&quot;,rr[rr.SIMP_CHINESE_FORMAL=41]=&quot;SIMP_CHINESE_FORMAL&quot;,rr[rr.SIMP_CHINESE_INFORMAL=42]=&quot;SIMP_CHINESE_INFORMAL&quot;,rr[rr.TAMIL=43]=&quot;TAMIL&quot;,rr[rr.TELUGU=44]=&quot;TELUGU&quot;,rr[rr.THAI=45]=&quot;THAI&quot;,rr[rr.TIBETAN=46]=&quot;TIBETAN&quot;,rr[rr.TRAD_CHINESE_FORMAL=47]=&quot;TRAD_CHINESE_FORMAL&quot;,rr[rr.TRAD_CHINESE_INFORMAL=48]=&quot;TRAD_CHINESE_INFORMAL&quot;,rr[rr.UPPER_ARMENIAN=49]=&quot;UPPER_ARMENIAN&quot;,rr[rr.DISCLOSURE_OPEN=50]=&quot;DISCLOSURE_OPEN&quot;;function nr(A){return{name:&quot;margin-&quot;+A,initialValue:&quot;0&quot;,prefix:!1,type:Ce.TOKEN_VALUE}}var sr,or,ir={name:&quot;list-style-type&quot;,initialValue:&quot;none&quot;,prefix:!(rr[rr.DISCLOSURE_CLOSED=51]=&quot;DISCLOSURE_CLOSED&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;disc&quot;:return tr.DISC;case&quot;circle&quot;:return tr.CIRCLE;case&quot;square&quot;:return tr.SQUARE;case&quot;decimal&quot;:return tr.DECIMAL;case&quot;cjk-decimal&quot;:return tr.CJK_DECIMAL;case&quot;decimal-leading-zero&quot;:return tr.DECIMAL_LEADING_ZERO;case&quot;lower-roman&quot;:return tr.LOWER_ROMAN;case&quot;upper-roman&quot;:return tr.UPPER_ROMAN;case&quot;lower-greek&quot;:return tr.LOWER_GREEK;case&quot;lower-alpha&quot;:return tr.LOWER_ALPHA;case&quot;upper-alpha&quot;:return tr.UPPER_ALPHA;case&quot;arabic-indic&quot;:return tr.ARABIC_INDIC;case&quot;armenian&quot;:return tr.ARMENIAN;case&quot;bengali&quot;:return tr.BENGALI;case&quot;cambodian&quot;:return tr.CAMBODIAN;case&quot;cjk-earthly-branch&quot;:return tr.CJK_EARTHLY_BRANCH;case&quot;cjk-heavenly-stem&quot;:return tr.CJK_HEAVENLY_STEM;case&quot;cjk-ideographic&quot;:return tr.CJK_IDEOGRAPHIC;case&quot;devanagari&quot;:return tr.DEVANAGARI;case&quot;ethiopic-numeric&quot;:return tr.ETHIOPIC_NUMERIC;case&quot;georgian&quot;:return tr.GEORGIAN;case&quot;gujarati&quot;:return tr.GUJARATI;case&quot;gurmukhi&quot;:return tr.GURMUKHI;case&quot;hebrew&quot;:return tr.HEBREW;case&quot;hiragana&quot;:return tr.HIRAGANA;case&quot;hiragana-iroha&quot;:return tr.HIRAGANA_IROHA;case&quot;japanese-formal&quot;:return tr.JAPANESE_FORMAL;case&quot;japanese-informal&quot;:return tr.JAPANESE_INFORMAL;case&quot;kannada&quot;:return tr.KANNADA;case&quot;katakana&quot;:return tr.KATAKANA;case&quot;katakana-iroha&quot;:return tr.KATAKANA_IROHA;case&quot;khmer&quot;:return tr.KHMER;case&quot;korean-hangul-formal&quot;:return tr.KOREAN_HANGUL_FORMAL;case&quot;korean-hanja-formal&quot;:return tr.KOREAN_HANJA_FORMAL;case&quot;korean-hanja-informal&quot;:return tr.KOREAN_HANJA_INFORMAL;case&quot;lao&quot;:return tr.LAO;case&quot;lower-armenian&quot;:return tr.LOWER_ARMENIAN;case&quot;malayalam&quot;:return tr.MALAYALAM;case&quot;mongolian&quot;:return tr.MONGOLIAN;case&quot;myanmar&quot;:return tr.MYANMAR;case&quot;oriya&quot;:return tr.ORIYA;case&quot;persian&quot;:return tr.PERSIAN;case&quot;simp-chinese-formal&quot;:return tr.SIMP_CHINESE_FORMAL;case&quot;simp-chinese-informal&quot;:return tr.SIMP_CHINESE_INFORMAL;case&quot;tamil&quot;:return tr.TAMIL;case&quot;telugu&quot;:return tr.TELUGU;case&quot;thai&quot;:return tr.THAI;case&quot;tibetan&quot;:return tr.TIBETAN;case&quot;trad-chinese-formal&quot;:return tr.TRAD_CHINESE_FORMAL;case&quot;trad-chinese-informal&quot;:return tr.TRAD_CHINESE_INFORMAL;case&quot;upper-armenian&quot;:return tr.UPPER_ARMENIAN;case&quot;disclosure-open&quot;:return tr.DISCLOSURE_OPEN;case&quot;disclosure-closed&quot;:return tr.DISCLOSURE_CLOSED;case&quot;none&quot;:default:return tr.NONE}}},ar=nr(&quot;top&quot;),cr=nr(&quot;right&quot;),Qr=nr(&quot;bottom&quot;),wr=nr(&quot;left&quot;);(or=sr||(sr={}))[or.VISIBLE=0]=&quot;VISIBLE&quot;,or[or.HIDDEN=1]=&quot;HIDDEN&quot;,or[or.SCROLL=2]=&quot;SCROLL&quot;;function ur(A){return{name:&quot;padding-&quot;+A,initialValue:&quot;0&quot;,prefix:!1,type:Ce.TYPE_VALUE,format:&quot;length-percentage&quot;}}var Ur,lr,Cr,gr,Er={name:&quot;overflow&quot;,initialValue:&quot;visible&quot;,prefix:!(or[or.AUTO=3]=&quot;AUTO&quot;),type:Ce.LIST,parse:function(A){return A.filter(zA).map(function(A){switch(A.value){case&quot;hidden&quot;:return sr.HIDDEN;case&quot;scroll&quot;:return sr.SCROLL;case&quot;auto&quot;:return sr.AUTO;case&quot;visible&quot;:default:return sr.VISIBLE}})}},Fr={name:&quot;overflow-wrap&quot;,initialValue:(lr=Ur||(Ur={})).NORMAL=&quot;normal&quot;,prefix:!(lr.BREAK_WORD=&quot;break-word&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;break-word&quot;:return Ur.BREAK_WORD;case&quot;normal&quot;:default:return Ur.NORMAL}}},hr=ur(&quot;top&quot;),Hr=ur(&quot;right&quot;),dr=ur(&quot;bottom&quot;),fr=ur(&quot;left&quot;);(gr=Cr||(Cr={}))[gr.LEFT=0]=&quot;LEFT&quot;,gr[gr.CENTER=1]=&quot;CENTER&quot;;var pr,Nr,Kr={name:&quot;text-align&quot;,initialValue:&quot;left&quot;,prefix:!(gr[gr.RIGHT=2]=&quot;RIGHT&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;right&quot;:return Cr.RIGHT;case&quot;center&quot;:case&quot;justify&quot;:return Cr.CENTER;case&quot;left&quot;:default:return Cr.LEFT}}};(Nr=pr||(pr={}))[Nr.STATIC=0]=&quot;STATIC&quot;,Nr[Nr.RELATIVE=1]=&quot;RELATIVE&quot;,Nr[Nr.ABSOLUTE=2]=&quot;ABSOLUTE&quot;,Nr[Nr.FIXED=3]=&quot;FIXED&quot;;var Ir,Tr,mr={name:&quot;position&quot;,initialValue:&quot;static&quot;,prefix:!(Nr[Nr.STICKY=4]=&quot;STICKY&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;relative&quot;:return pr.RELATIVE;case&quot;absolute&quot;:return pr.ABSOLUTE;case&quot;fixed&quot;:return pr.FIXED;case&quot;sticky&quot;:return pr.STICKY}return pr.STATIC}},Rr={name:&quot;text-shadow&quot;,initialValue:&quot;none&quot;,type:Ce.LIST,prefix:!1,parse:function(A){return 1===A.length&&JA(A[0],&quot;none&quot;)?[]:WA(A).map(function(A){for(var e={color:He.TRANSPARENT,offsetX:se,offsetY:se,blur:se},t=0,r=0;r<A.length;r++){var B=A[r];YA(B)?(0===t?e.offsetX=B:1===t?e.offsetY=B:e.blur=B,t++):e.color=we(B)}return e})}};(Tr=Ir||(Ir={}))[Tr.NONE=0]=&quot;NONE&quot;,Tr[Tr.LOWERCASE=1]=&quot;LOWERCASE&quot;,Tr[Tr.UPPERCASE=2]=&quot;UPPERCASE&quot;;var Lr,Or,vr={name:&quot;text-transform&quot;,initialValue:&quot;none&quot;,prefix:!(Tr[Tr.CAPITALIZE=3]=&quot;CAPITALIZE&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;uppercase&quot;:return Ir.UPPERCASE;case&quot;lowercase&quot;:return Ir.LOWERCASE;case&quot;capitalize&quot;:return Ir.CAPITALIZE}return Ir.NONE}},Dr={name:&quot;transform&quot;,initialValue:&quot;none&quot;,prefix:!0,type:Ce.VALUE,parse:function(A){if(A.type===sA.IDENT_TOKEN&&&quot;none&quot;===A.value)return null;if(A.type!==sA.FUNCTION)return null;var e=Sr[A.name];if(void 0===e)throw new Error('Attempting to parse an unsupported transform function &quot;'+A.name+'&quot;');return e(A.values)}},Sr={matrix:function(A){var e=A.filter(function(A){return A.type===sA.NUMBER_TOKEN}).map(function(A){return A.number});return 6===e.length?e:null},matrix3d:function(A){var e=A.filter(function(A){return A.type===sA.NUMBER_TOKEN}).map(function(A){return A.number}),t=e[0],r=e[1],B=(e[2],e[3],e[4]),n=e[5],s=(e[6],e[7],e[8],e[9],e[10],e[11],e[12]),o=e[13];e[14],e[15];return 16===e.length?[t,r,B,n,s,o]:null}},br={type:sA.PERCENTAGE_TOKEN,number:50,flags:4},Mr=[br,br],yr={name:&quot;transform-origin&quot;,initialValue:&quot;50% 50%&quot;,prefix:!0,type:Ce.LIST,parse:function(A){var e=A.filter(qA);return 2!==e.length?Mr:[e[0],e[1]]}};(Or=Lr||(Lr={}))[Or.VISIBLE=0]=&quot;VISIBLE&quot;,Or[Or.HIDDEN=1]=&quot;HIDDEN&quot;;var _r,Pr,xr={name:&quot;visible&quot;,initialValue:&quot;none&quot;,prefix:!(Or[Or.COLLAPSE=2]=&quot;COLLAPSE&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;hidden&quot;:return Lr.HIDDEN;case&quot;collapse&quot;:return Lr.COLLAPSE;case&quot;visible&quot;:default:return Lr.VISIBLE}}};(Pr=_r||(_r={})).NORMAL=&quot;normal&quot;,Pr.BREAK_ALL=&quot;break-all&quot;;var Vr,zr,Xr={name:&quot;word-break&quot;,initialValue:&quot;normal&quot;,prefix:!(Pr.KEEP_ALL=&quot;keep-all&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;break-all&quot;:return _r.BREAK_ALL;case&quot;keep-all&quot;:return _r.KEEP_ALL;case&quot;normal&quot;:default:return _r.NORMAL}}},Jr={name:&quot;z-index&quot;,initialValue:&quot;auto&quot;,prefix:!1,type:Ce.VALUE,parse:function(A){if(A.type===sA.IDENT_TOKEN)return{auto:!0,order:0};if(VA(A))return{auto:!1,order:A.number};throw new Error(&quot;Invalid z-index number parsed&quot;)}},Gr={name:&quot;opacity&quot;,initialValue:&quot;1&quot;,type:Ce.VALUE,prefix:!1,parse:function(A){return VA(A)?A.number:1}},kr={name:&quot;text-decoration-color&quot;,initialValue:&quot;transparent&quot;,prefix:!1,type:Ce.TYPE_VALUE,format:&quot;color&quot;},Wr={name:&quot;text-decoration-line&quot;,initialValue:&quot;none&quot;,prefix:!1,type:Ce.LIST,parse:function(A){return A.filter(zA).map(function(A){switch(A.value){case&quot;underline&quot;:return 1;case&quot;overline&quot;:return 2;case&quot;line-through&quot;:return 3;case&quot;none&quot;:return 4}return 0}).filter(function(A){return 0!==A})}},Yr={name:&quot;font-family&quot;,initialValue:&quot;&quot;,prefix:!1,type:Ce.LIST,parse:function(A){return A.filter(qr).map(function(A){return A.value})}},qr=function(A){return A.type===sA.STRING_TOKEN||A.type===sA.IDENT_TOKEN},Zr={name:&quot;font-size&quot;,initialValue:&quot;0&quot;,prefix:!1,type:Ce.TYPE_VALUE,format:&quot;length&quot;},jr={name:&quot;font-weight&quot;,initialValue:&quot;normal&quot;,type:Ce.VALUE,prefix:!1,parse:function(A){if(VA(A))return A.number;if(zA(A))switch(A.value){case&quot;bold&quot;:return 700;case&quot;normal&quot;:default:return 400}return 400}},$r={name:&quot;font-variant&quot;,initialValue:&quot;none&quot;,type:Ce.LIST,prefix:!1,parse:function(A){return A.filter(zA).map(function(A){return A.value})}};(zr=Vr||(Vr={})).NORMAL=&quot;normal&quot;,zr.ITALIC=&quot;italic&quot;;function AB(A,e){return 0!=(A&e)}function eB(A,e,t){if(!A)return&quot;&quot;;var r=A[Math.min(e,A.length-1)];return r?t?r.open:r.close:&quot;&quot;}var tB={name:&quot;font-style&quot;,initialValue:&quot;normal&quot;,prefix:!(zr.OBLIQUE=&quot;oblique&quot;),type:Ce.IDENT_VALUE,parse:function(A){switch(A){case&quot;oblique&quot;:return Vr.OBLIQUE;case&quot;italic&quot;:return Vr.ITALIC;case&quot;normal&quot;:default:return Vr.NORMAL}}},rB={name:&quot;content&quot;,initialValue:&quot;none&quot;,type:Ce.LIST,prefix:!1,parse:function(A){if(0===A.length)return[];var e=A[0];return e.type===sA.IDENT_TOKEN&&&quot;none&quot;===e.value?[]:A}},BB={name:&quot;counter-increment&quot;,initialValue:&quot;none&quot;,prefix:!0,type:Ce.LIST,parse:function(A){if(0===A.length)return null;var e=A[0];if(e.type===sA.IDENT_TOKEN&&&quot;none&quot;===e.value)return null;for(var t=[],r=A.filter(GA),B=0;B<r.length;B++){var n=r[B],s=r[B+1];if(n.type===sA.IDENT_TOKEN){var o=s&&VA(s)?s.number:1;t.push({counter:n.value,increment:o})}}return t}},nB={name:&quot;counter-reset&quot;,initialValue:&quot;none&quot;,prefix:!0,type:Ce.LIST,parse:function(A){if(0===A.length)return[];for(var e=[],t=A.filter(GA),r=0;r<t.length;r++){var B=t[r],n=t[r+1];if(zA(B)&&&quot;none&quot;!==B.value){var s=n&&VA(n)?n.number:0;e.push({counter:B.value,reset:s})}}return e}},sB={name:&quot;quotes&quot;,initialValue:&quot;none&quot;,prefix:!0,type:Ce.LIST,parse:function(A){if(0===A.length)return null;var e=A[0];if(e.type===sA.IDENT_TOKEN&&&quot;none&quot;===e.value)return null;var t=[],r=A.filter(XA);if(r.length%2!=0)return null;for(var B=0;B<r.length;B+=2){var n=r[B].value,s=r[B+1].value;t.push({open:n,close:s})}return t}},oB={name:&quot;box-shadow&quot;,initialValue:&quot;none&quot;,type:Ce.LIST,prefix:!1,parse:function(A){return 1===A.length&&JA(A[0],&quot;none&quot;)?[]:WA(A).map(function(A){for(var e={color:255,offsetX:se,offsetY:se,blur:se,spread:se,inset:!1},t=0,r=0;r<A.length;r++){var B=A[r];JA(B,&quot;inset&quot;)?e.inset=!0:YA(B)?(0===t?e.offsetX=B:1===t?e.offsetY=B:2===t?e.blur=B:e.spread=B,t++):e.color=we(B)}return e})}},iB=(aB.prototype.isVisible=function(){return 0<this.display&&0<this.opacity&&this.visibility===Lr.VISIBLE},aB.prototype.isTransparent=function(){return ee(this.backgroundColor)},aB.prototype.isTransformed=function(){return null!==this.transform},aB.prototype.isPositioned=function(){return this.position!==pr.STATIC},aB.prototype.isPositionedWithZIndex=function(){return this.isPositioned()&&!this.zIndex.auto},aB.prototype.isFloating=function(){return this.float!==Dt.NONE},aB.prototype.isInlineLevel=function(){return AB(this.display,4)||AB(this.display,33554432)||AB(this.display,268435456)||AB(this.display,536870912)||AB(this.display,67108864)||AB(this.display,134217728)},aB);function aB(A){this.backgroundClip=uB(me,A.backgroundClip),this.backgroundColor=uB(Re,A.backgroundColor),this.backgroundImage=uB(Qt,A.backgroundImage),this.backgroundOrigin=uB(wt,A.backgroundOrigin),this.backgroundPosition=uB(ut,A.backgroundPosition),this.backgroundRepeat=uB(Ct,A.backgroundRepeat),this.backgroundSize=uB(dt,A.backgroundSize),this.borderTopColor=uB(pt,A.borderTopColor),this.borderRightColor=uB(Nt,A.borderRightColor),this.borderBottomColor=uB(Kt,A.borderBottomColor),this.borderLeftColor=uB(It,A.borderLeftColor),this.borderTopLeftRadius=uB(Tt,A.borderTopLeftRadius),this.borderTopRightRadius=uB(mt,A.borderTopRightRadius),this.borderBottomRightRadius=uB(Rt,A.borderBottomRightRadius),this.borderBottomLeftRadius=uB(Lt,A.borderBottomLeftRadius),this.borderTopStyle=uB(bt,A.borderTopStyle),this.borderRightStyle=uB(Mt,A.borderRightStyle),this.borderBottomStyle=uB(yt,A.borderBottomStyle),this.borderLeftStyle=uB(_t,A.borderLeftStyle),this.borderTopWidth=uB(Pt,A.borderTopWidth),this.borderRightWidth=uB(xt,A.borderRightWidth),this.borderBottomWidth=uB(Vt,A.borderBottomWidth),this.borderLeftWidth=uB(zt,A.borderLeftWidth),this.boxShadow=uB(oB,A.boxShadow),this.color=uB(Xt,A.color),this.display=uB(Jt,A.display),this.float=uB(Zt,A.cssFloat),this.fontFamily=uB(Yr,A.fontFamily),this.fontSize=uB(Zr,A.fontSize),this.fontStyle=uB(tB,A.fontStyle),this.fontVariant=uB($r,A.fontVariant),this.fontWeight=uB(jr,A.fontWeight),this.letterSpacing=uB(jt,A.letterSpacing),this.lineBreak=uB($t,A.lineBreak),this.lineHeight=uB(Ar,A.lineHeight),this.listStyleImage=uB(er,A.listStyleImage),this.listStylePosition=uB(Br,A.listStylePosition),this.listStyleType=uB(ir,A.listStyleType),this.marginTop=uB(ar,A.marginTop),this.marginRight=uB(cr,A.marginRight),this.marginBottom=uB(Qr,A.marginBottom),this.marginLeft=uB(wr,A.marginLeft),this.opacity=uB(Gr,A.opacity);var e=uB(Er,A.overflow);this.overflowX=e[0],this.overflowY=e[1<e.length?1:0],this.overflowWrap=uB(Fr,A.overflowWrap),this.paddingTop=uB(hr,A.paddingTop),this.paddingRight=uB(Hr,A.paddingRight),this.paddingBottom=uB(dr,A.paddingBottom),this.paddingLeft=uB(fr,A.paddingLeft),this.position=uB(mr,A.position),this.textAlign=uB(Kr,A.textAlign),this.textDecorationColor=uB(kr,A.textDecorationColor||A.color),this.textDecorationLine=uB(Wr,A.textDecorationLine),this.textShadow=uB(Rr,A.textShadow),this.textTransform=uB(vr,A.textTransform),this.transform=uB(Dr,A.transform),this.transformOrigin=uB(yr,A.transformOrigin),this.visibility=uB(xr,A.visibility),this.wordBreak=uB(Xr,A.wordBreak),this.zIndex=uB(Jr,A.zIndex)}var cB,QB=function(A){this.content=uB(rB,A.content),this.quotes=uB(sB,A.quotes)},wB=function(A){this.counterIncrement=uB(BB,A.counterIncrement),this.counterReset=uB(nB,A.counterReset)},uB=function(A,e){var t=new MA,r=null!=e?e.toString():A.initialValue;t.write(r);var B=new _A(t.read());switch(A.type){case Ce.IDENT_VALUE:var n=B.parseComponentValue();return A.parse(zA(n)?n.value:A.initialValue);case Ce.VALUE:return A.parse(B.parseComponentValue());case Ce.LIST:return A.parse(B.parseComponentValues());case Ce.TOKEN_VALUE:return B.parseComponentValue();case Ce.TYPE_VALUE:switch(A.format){case&quot;angle&quot;:return ce(B.parseComponentValue());case&quot;color&quot;:return we(B.parseComponentValue());case&quot;image&quot;:return at(B.parseComponentValue());case&quot;length&quot;:var s=B.parseComponentValue();return YA(s)?s:se;case&quot;length-percentage&quot;:var o=B.parseComponentValue();return qA(o)?o:se}}throw new Error(&quot;Attempting to parse unsupported css format type &quot;+A.format)},UB=function(A){this.styles=new iB(window.getComputedStyle(A,null)),this.textNodes=[],this.elements=[],null!==this.styles.transform&&un(A)&&(A.style.transform=&quot;none&quot;),this.bounds=T(A),this.flags=0},lB=function(A,e){this.text=A,this.bounds=e},CB=function(A){var e=A.ownerDocument;if(e){var t=e.createElement(&quot;html2canvaswrapper&quot;);t.appendChild(A.cloneNode(!0));var r=A.parentNode;if(r){r.replaceChild(t,A);var B=T(t);return t.firstChild&&r.replaceChild(t.firstChild,t),B}}return new I(0,0,0,0)},gB=function(A,e,t){var r=A.ownerDocument;if(!r)throw new Error(&quot;Node has no owner document&quot;);var B=r.createRange();return B.setStart(A,e),B.setEnd(A,e+t),I.fromClientRect(B.getBoundingClientRect())},EB=function(A,e){return 0!==e.letterSpacing?c(A).map(function(A){return l(A)}):FB(A,e)},FB=function(A,e){for(var t,r=function(A,e){var t=c(A),r=u(t,e),B=r[0],n=r[1],s=r[2],o=t.length,i=0,a=0;return{next:function(){if(o<=a)return{done:!0,value:null};for(var A=Y;a<o&&(A=w(t,n,B,++a,s))===Y;);if(A===Y&&a!==o)return{done:!0,value:null};var e=new BA(t,A,i,a);return i=a,{value:e,done:!1}}}}(A,{lineBreak:e.lineBreak,wordBreak:e.overflowWrap===Ur.BREAK_WORD?&quot;break-word&quot;:e.wordBreak}),B=[];!(t=r.next()).done;)t.value&&B.push(t.value.slice());return B},hB=function(A,e){this.text=HB(A.data,e.textTransform),this.textBounds=function(A,t,r){var e=EB(A,t),B=[],n=0;return e.forEach(function(A){if(t.textDecorationLine.length||0<A.trim().length)if(ve.SUPPORT_RANGE_BOUNDS)B.push(new lB(A,gB(r,n,A.length)));else{var e=r.splitText(A.length);B.push(new lB(A,CB(r))),r=e}else ve.SUPPORT_RANGE_BOUNDS||(r=r.splitText(A.length));n+=A.length}),B}(this.text,e,A)},HB=function(A,e){switch(e){case Ir.LOWERCASE:return A.toLowerCase();case Ir.CAPITALIZE:return A.replace(dB,fB);case Ir.UPPERCASE:return A.toUpperCase();default:return A}},dB=/(^|\s|:|-|\(|\))([a-z])/g,fB=function(A,e,t){return 0<A.length?e+t.toUpperCase():A},pB=(A(NB,cB=UB),NB);function NB(A){var e=cB.call(this,A)||this;return e.src=A.currentSrc||A.src,e.intrinsicWidth=A.naturalWidth,e.intrinsicHeight=A.naturalHeight,be.getInstance().addImage(e.src),e}var KB,IB=(A(TB,KB=UB),TB);function TB(A){var e=KB.call(this,A)||this;return e.canvas=A,e.intrinsicWidth=A.width,e.intrinsicHeight=A.height,e}var mB,RB=(A(LB,mB=UB),LB);function LB(A){var e=mB.call(this,A)||this,t=new XMLSerializer;return e.svg=&quot;data:image/svg+xml,&quot;+encodeURIComponent(t.serializeToString(A)),e.intrinsicWidth=A.width.baseVal.value,e.intrinsicHeight=A.height.baseVal.value,be.getInstance().addImage(e.svg),e}var OB,vB=(A(DB,OB=UB),DB);function DB(A){var e=OB.call(this,A)||this;return e.value=A.value,e}var SB,bB=(A(MB,SB=UB),MB);function MB(A){var e=SB.call(this,A)||this;return e.start=A.start,e.reversed=&quot;boolean&quot;==typeof A.reversed&&!0===A.reversed,e}var yB,_B=[{type:sA.DIMENSION_TOKEN,flags:0,unit:&quot;px&quot;,number:3}],PB=[{type:sA.PERCENTAGE_TOKEN,flags:0,number:50}],xB=&quot;checkbox&quot;,VB=&quot;radio&quot;,zB=&quot;password&quot;,XB=707406591,JB=(A(GB,yB=UB),GB);function GB(A){var e=yB.call(this,A)||this;switch(e.type=A.type.toLowerCase(),e.checked=A.checked,e.value=function(A){var e=A.type===zB?new Array(A.value.length+1).join(&quot;•&quot;):A.value;return 0===e.length?A.placeholder||&quot;&quot;:e}(A),e.type!==xB&&e.type!==VB||(e.styles.backgroundColor=3739148031,e.styles.borderTopColor=e.styles.borderRightColor=e.styles.borderBottomColor=e.styles.borderLeftColor=2779096575,e.styles.borderTopWidth=e.styles.borderRightWidth=e.styles.borderBottomWidth=e.styles.borderLeftWidth=1,e.styles.borderTopStyle=e.styles.borderRightStyle=e.styles.borderBottomStyle=e.styles.borderLeftStyle=ht.SOLID,e.styles.backgroundClip=[Ee.BORDER_BOX],e.styles.backgroundOrigin=[0],e.bounds=function(A){return A.width>A.height?new I(A.left+(A.width-A.height)/2,A.top,A.height,A.height):A.width<A.height?new I(A.left,A.top+(A.height-A.width)/2,A.width,A.width):A}(e.bounds)),e.type){case xB:e.styles.borderTopRightRadius=e.styles.borderTopLeftRadius=e.styles.borderBottomRightRadius=e.styles.borderBottomLeftRadius=_B;break;case VB:e.styles.borderTopRightRadius=e.styles.borderTopLeftRadius=e.styles.borderBottomRightRadius=e.styles.borderBottomLeftRadius=PB}return e}var kB,WB=(A(YB,kB=UB),YB);function YB(A){var e=kB.call(this,A)||this,t=A.options[A.selectedIndex||0];return e.value=t&&t.text||&quot;&quot;,e}var qB,ZB=(A(jB,qB=UB),jB);function jB(A){var e=qB.call(this,A)||this;return e.value=A.value,e}function $B(A){return we(_A.create(A).parseComponentValue())}var An,en=(A(tn,An=UB),tn);function tn(A){var e=An.call(this,A)||this;e.src=A.src,e.width=parseInt(A.width,10),e.height=parseInt(A.height,10),e.backgroundColor=e.styles.backgroundColor;try{if(A.contentWindow&&A.contentWindow.document&&A.contentWindow.document.documentElement){e.tree=on(A.contentWindow.document.documentElement);var t=A.contentWindow.document.documentElement?$B(getComputedStyle(A.contentWindow.document.documentElement).backgroundColor):He.TRANSPARENT,r=A.contentWindow.document.body?$B(getComputedStyle(A.contentWindow.document.body).backgroundColor):He.TRANSPARENT;e.backgroundColor=ee(t)?ee(r)?e.styles.backgroundColor:r:t}}catch(A){}return e}function rn(A){return&quot;STYLE&quot;===A.tagName}var Bn=[&quot;OL&quot;,&quot;UL&quot;,&quot;MENU&quot;],nn=function(A,e,t){for(var r=A.firstChild,B=void 0;r;r=B)if(B=r.nextSibling,Qn(r)&&0<r.data.trim().length)e.textNodes.push(new hB(r,e.styles));else if(wn(r)){var n=sn(r);n.styles.isVisible()&&(an(r,n,t)?n.flags|=4:cn(n.styles)&&(n.flags|=2),-1!==Bn.indexOf(r.tagName)&&(n.flags|=8),e.elements.push(n),dn(r)||gn(r)||fn(r)||nn(r,n,t))}},sn=function(A){return hn(A)?new pB(A):Fn(A)?new IB(A):gn(A)?new RB(A):Un(A)?new vB(A):ln(A)?new bB(A):Cn(A)?new JB(A):fn(A)?new WB(A):dn(A)?new ZB(A):Hn(A)?new en(A):new UB(A)},on=function(A){var e=sn(A);return e.flags|=4,nn(A,e,e),e},an=function(A,e,t){return e.styles.isPositionedWithZIndex()||e.styles.opacity<1||e.styles.isTransformed()||En(A)&&t.styles.isTransparent()},cn=function(A){return A.isPositioned()||A.isFloating()},Qn=function(A){return A.nodeType===Node.TEXT_NODE},wn=function(A){return A.nodeType===Node.ELEMENT_NODE},un=function(A){return void 0!==A.style},Un=function(A){return&quot;LI&quot;===A.tagName},ln=function(A){return&quot;OL&quot;===A.tagName},Cn=function(A){return&quot;INPUT&quot;===A.tagName},gn=function(A){return&quot;svg&quot;===A.tagName},En=function(A){return&quot;BODY&quot;===A.tagName},Fn=function(A){return&quot;CANVAS&quot;===A.tagName},hn=function(A){return&quot;IMG&quot;===A.tagName},Hn=function(A){return&quot;IFRAME&quot;===A.tagName},dn=function(A){return&quot;TEXTAREA&quot;===A.tagName},fn=function(A){return&quot;SELECT&quot;===A.tagName},pn=(Nn.prototype.getCounterValue=function(A){var e=this.counters[A];return e&&e.length?e[e.length-1]:1},Nn.prototype.getCounterValues=function(A){var e=this.counters[A];return e||[]},Nn.prototype.pop=function(A){var e=this;A.forEach(function(A){return e.counters[A].pop()})},Nn.prototype.parse=function(A){var t=this,e=A.counterIncrement,r=A.counterReset;null!==e&&e.forEach(function(A){var e=t.counters[A.counter];e&&(e[Math.max(0,e.length-1)]+=A.increment)});var B=[];return r.forEach(function(A){var e=t.counters[A.counter];B.push(A.counter),e||(e=t.counters[A.counter]=[]),e.push(A.reset)}),B},Nn);function Nn(){this.counters={}}function Kn(r,A,e,B,t,n){return r<A||e<r?yn(r,t,0<n.length):B.integers.reduce(function(A,e,t){for(;e<=r;)r-=e,A+=B.values[t];return A},&quot;&quot;)+n}function In(A,e,t,r){for(var B=&quot;&quot;;t||A--,B=r(A)+B,e<=(A/=e)*e;);return B}function Tn(A,e,t,r,B){var n=t-e+1;return(A<0?&quot;-&quot;:&quot;&quot;)+(In(Math.abs(A),n,r,function(A){return l(Math.floor(A%n)+e)})+B)}function mn(A,e,t){void 0===t&&(t=&quot;. &quot;);var r=e.length;return In(Math.abs(A),r,!1,function(A){return e[Math.floor(A%r)]})+t}function Rn(A,e,t,r,B,n){if(A<-9999||9999<A)return yn(A,tr.CJK_DECIMAL,0<B.length);var s=Math.abs(A),o=B;if(0===s)return e[0]+o;for(var i=0;0<s&&i<=4;i++){var a=s%10;0==a&&AB(n,1)&&&quot;&quot;!==o?o=e[a]+o:1<a||1==a&&0===i||1==a&&1===i&&AB(n,2)||1==a&&1===i&&AB(n,4)&&100<A||1==a&&1<i&&AB(n,8)?o=e[a]+(0<i?t[i-1]:&quot;&quot;)+o:1==a&&0<i&&(o=t[i-1]+o),s=Math.floor(s/10)}return(A<0?r:&quot;&quot;)+o}var Ln,On,vn={integers:[1e3,900,500,400,100,90,50,40,10,9,5,4,1],values:[&quot;M&quot;,&quot;CM&quot;,&quot;D&quot;,&quot;CD&quot;,&quot;C&quot;,&quot;XC&quot;,&quot;L&quot;,&quot;XL&quot;,&quot;X&quot;,&quot;IX&quot;,&quot;V&quot;,&quot;IV&quot;,&quot;I&quot;]},Dn={integers:[9e3,8e3,7e3,6e3,5e3,4e3,3e3,2e3,1e3,900,800,700,600,500,400,300,200,100,90,80,70,60,50,40,30,20,10,9,8,7,6,5,4,3,2,1],values:[&quot;Ք&quot;,&quot;Փ&quot;,&quot;Ւ&quot;,&quot;Ց&quot;,&quot;Ր&quot;,&quot;Տ&quot;,&quot;Վ&quot;,&quot;Ս&quot;,&quot;Ռ&quot;,&quot;Ջ&quot;,&quot;Պ&quot;,&quot;Չ&quot;,&quot;Ո&quot;,&quot;Շ&quot;,&quot;Ն&quot;,&quot;Յ&quot;,&quot;Մ&quot;,&quot;Ճ&quot;,&quot;Ղ&quot;,&quot;Ձ&quot;,&quot;Հ&quot;,&quot;Կ&quot;,&quot;Ծ&quot;,&quot;Խ&quot;,&quot;Լ&quot;,&quot;Ի&quot;,&quot;Ժ&quot;,&quot;Թ&quot;,&quot;Ը&quot;,&quot;Է&quot;,&quot;Զ&quot;,&quot;Ե&quot;,&quot;Դ&quot;,&quot;Գ&quot;,&quot;Բ&quot;,&quot;Ա&quot;]},Sn={integers:[1e4,9e3,8e3,7e3,6e3,5e3,4e3,3e3,2e3,1e3,400,300,200,100,90,80,70,60,50,40,30,20,19,18,17,16,15,10,9,8,7,6,5,4,3,2,1],values:[&quot;י׳&quot;,&quot;ט׳&quot;,&quot;ח׳&quot;,&quot;ז׳&quot;,&quot;ו׳&quot;,&quot;ה׳&quot;,&quot;ד׳&quot;,&quot;ג׳&quot;,&quot;ב׳&quot;,&quot;א׳&quot;,&quot;ת&quot;,&quot;ש&quot;,&quot;ר&quot;,&quot;ק&quot;,&quot;צ&quot;,&quot;פ&quot;,&quot;ע&quot;,&quot;ס&quot;,&quot;נ&quot;,&quot;מ&quot;,&quot;ל&quot;,&quot;כ&quot;,&quot;יט&quot;,&quot;יח&quot;,&quot;יז&quot;,&quot;טז&quot;,&quot;טו&quot;,&quot;י&quot;,&quot;ט&quot;,&quot;ח&quot;,&quot;ז&quot;,&quot;ו&quot;,&quot;ה&quot;,&quot;ד&quot;,&quot;ג&quot;,&quot;ב&quot;,&quot;א&quot;]},bn={integers:[1e4,9e3,8e3,7e3,6e3,5e3,4e3,3e3,2e3,1e3,900,800,700,600,500,400,300,200,100,90,80,70,60,50,40,30,20,10,9,8,7,6,5,4,3,2,1],values:[&quot;ჵ&quot;,&quot;ჰ&quot;,&quot;ჯ&quot;,&quot;ჴ&quot;,&quot;ხ&quot;,&quot;ჭ&quot;,&quot;წ&quot;,&quot;ძ&quot;,&quot;ც&quot;,&quot;ჩ&quot;,&quot;შ&quot;,&quot;ყ&quot;,&quot;ღ&quot;,&quot;ქ&quot;,&quot;ფ&quot;,&quot;ჳ&quot;,&quot;ტ&quot;,&quot;ს&quot;,&quot;რ&quot;,&quot;ჟ&quot;,&quot;პ&quot;,&quot;ო&quot;,&quot;ჲ&quot;,&quot;ნ&quot;,&quot;მ&quot;,&quot;ლ&quot;,&quot;კ&quot;,&quot;ი&quot;,&quot;თ&quot;,&quot;ჱ&quot;,&quot;ზ&quot;,&quot;ვ&quot;,&quot;ე&quot;,&quot;დ&quot;,&quot;გ&quot;,&quot;ბ&quot;,&quot;ა&quot;]},Mn=&quot;마이너스&quot;,yn=function(A,e,t){var r=t?&quot;. &quot;:&quot;&quot;,B=t?&quot;、&quot;:&quot;&quot;,n=t?&quot;, &quot;:&quot;&quot;,s=t?&quot; &quot;:&quot;&quot;;switch(e){case tr.DISC:return&quot;•&quot;+s;case tr.CIRCLE:return&quot;◦&quot;+s;case tr.SQUARE:return&quot;◾&quot;+s;case tr.DECIMAL_LEADING_ZERO:var o=Tn(A,48,57,!0,r);return o.length<4?&quot;0&quot;+o:o;case tr.CJK_DECIMAL:return mn(A,&quot;〇一二三四五六七八九&quot;,B);case tr.LOWER_ROMAN:return Kn(A,1,3999,vn,tr.DECIMAL,r).toLowerCase();case tr.UPPER_ROMAN:return Kn(A,1,3999,vn,tr.DECIMAL,r);case tr.LOWER_GREEK:return Tn(A,945,969,!1,r);case tr.LOWER_ALPHA:return Tn(A,97,122,!1,r);case tr.UPPER_ALPHA:return Tn(A,65,90,!1,r);case tr.ARABIC_INDIC:return Tn(A,1632,1641,!0,r);case tr.ARMENIAN:case tr.UPPER_ARMENIAN:return Kn(A,1,9999,Dn,tr.DECIMAL,r);case tr.LOWER_ARMENIAN:return Kn(A,1,9999,Dn,tr.DECIMAL,r).toLowerCase();case tr.BENGALI:return Tn(A,2534,2543,!0,r);case tr.CAMBODIAN:case tr.KHMER:return Tn(A,6112,6121,!0,r);case tr.CJK_EARTHLY_BRANCH:return mn(A,&quot;子丑寅卯辰巳午未申酉戌亥&quot;,B);case tr.CJK_HEAVENLY_STEM:return mn(A,&quot;甲乙丙丁戊己庚辛壬癸&quot;,B);case tr.CJK_IDEOGRAPHIC:case tr.TRAD_CHINESE_INFORMAL:return Rn(A,&quot;零一二三四五六七八九&quot;,&quot;十百千萬&quot;,&quot;負&quot;,B,14);case tr.TRAD_CHINESE_FORMAL:return Rn(A,&quot;零壹貳參肆伍陸柒捌玖&quot;,&quot;拾佰仟萬&quot;,&quot;負&quot;,B,15);case tr.SIMP_CHINESE_INFORMAL:return Rn(A,&quot;零一二三四五六七八九&quot;,&quot;十百千萬&quot;,&quot;负&quot;,B,14);case tr.SIMP_CHINESE_FORMAL:return Rn(A,&quot;零壹贰叁肆伍陆柒捌玖&quot;,&quot;拾佰仟萬&quot;,&quot;负&quot;,B,15);case tr.JAPANESE_INFORMAL:return Rn(A,&quot;〇一二三四五六七八九&quot;,&quot;十百千万&quot;,&quot;マイナス&quot;,B,0);case tr.JAPANESE_FORMAL:return Rn(A,&quot;零壱弐参四伍六七八九&quot;,&quot;拾百千万&quot;,&quot;マイナス&quot;,B,7);case tr.KOREAN_HANGUL_FORMAL:return Rn(A,&quot;영일이삼사오육칠팔구&quot;,&quot;십백천만&quot;,Mn,n,7);case tr.KOREAN_HANJA_INFORMAL:return Rn(A,&quot;零一二三四五六七八九&quot;,&quot;十百千萬&quot;,Mn,n,0);case tr.KOREAN_HANJA_FORMAL:return Rn(A,&quot;零壹貳參四五六七八九&quot;,&quot;拾百千&quot;,Mn,n,7);case tr.DEVANAGARI:return Tn(A,2406,2415,!0,r);case tr.GEORGIAN:return Kn(A,1,19999,bn,tr.DECIMAL,r);case tr.GUJARATI:return Tn(A,2790,2799,!0,r);case tr.GURMUKHI:return Tn(A,2662,2671,!0,r);case tr.HEBREW:return Kn(A,1,10999,Sn,tr.DECIMAL,r);case tr.HIRAGANA:return mn(A,&quot;あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑをん&quot;);case tr.HIRAGANA_IROHA:return mn(A,&quot;いろはにほへとちりぬるをわかよたれそつねならむうゐのおくやまけふこえてあさきゆめみしゑひもせす&quot;);case tr.KANNADA:return Tn(A,3302,3311,!0,r);case tr.KATAKANA:return mn(A,&quot;アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヰヱヲン&quot;,B);case tr.KATAKANA_IROHA:return mn(A,&quot;イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセス&quot;,B);case tr.LAO:return Tn(A,3792,3801,!0,r);case tr.MONGOLIAN:return Tn(A,6160,6169,!0,r);case tr.MYANMAR:return Tn(A,4160,4169,!0,r);case tr.ORIYA:return Tn(A,2918,2927,!0,r);case tr.PERSIAN:return Tn(A,1776,1785,!0,r);case tr.TAMIL:return Tn(A,3046,3055,!0,r);case tr.TELUGU:return Tn(A,3174,3183,!0,r);case tr.THAI:return Tn(A,3664,3673,!0,r);case tr.TIBETAN:return Tn(A,3872,3881,!0,r);case tr.DECIMAL:default:return Tn(A,48,57,!0,r)}},_n=&quot;data-html2canvas-ignore&quot;,Pn=(xn.prototype.toIFrame=function(A,e){var t=this,r=Xn(A,e);if(!r.contentWindow)return Promise.reject(&quot;Unable to find iframe window&quot;);var B=A.defaultView.pageXOffset,n=A.defaultView.pageYOffset,s=r.contentWindow,o=s.document,i=Jn(r).then(function(){t.scrolledElements.forEach(Yn),s&&(s.scrollTo(e.left,e.top),!/(iPad|iPhone|iPod)/g.test(navigator.userAgent)||s.scrollY===e.top&&s.scrollX===e.left||(o.documentElement.style.top=-e.top+&quot;px&quot;,o.documentElement.style.left=-e.left+&quot;px&quot;,o.documentElement.style.position=&quot;absolute&quot;));var A=t.options.onclone;return void 0===t.clonedReferenceElement?Promise.reject(&quot;Error finding the &quot;+t.referenceElement.nodeName+&quot; in the cloned document&quot;):&quot;function&quot;==typeof A?Promise.resolve().then(function(){return A(o)}).then(function(){return r}):r});return o.open(),o.write(kn(document.doctype)+&quot;<html></html>&quot;),Wn(this.referenceElement.ownerDocument,B,n),o.replaceChild(o.adoptNode(this.documentElement),o.documentElement),o.close(),i},xn.prototype.createElementClone=function(A){return Fn(A)?this.createCanvasClone(A):rn(A)?this.createStyleClone(A):A.cloneNode(!1)},xn.prototype.createStyleClone=function(A){try{var e=A.sheet;if(e&&e.cssRules){var t=[].slice.call(e.cssRules,0).reduce(function(A,e){return e&&&quot;string&quot;==typeof e.cssText?A+e.cssText:A},&quot;&quot;),r=A.cloneNode(!1);return r.textContent=t,r}}catch(A){if(De.getInstance(this.options.id).error(&quot;Unable to access cssRules property&quot;,A),&quot;SecurityError&quot;!==A.name)throw A}return A.cloneNode(!1)},xn.prototype.createCanvasClone=function(A){if(this.options.inlineImages&&A.ownerDocument){var e=A.ownerDocument.createElement(&quot;img&quot;);try{return e.src=A.toDataURL(),e}catch(A){De.getInstance(this.options.id).info(&quot;Unable to clone canvas contents, canvas is tainted&quot;)}}var t=A.cloneNode(!1);try{t.width=A.width,t.height=A.height;var r=A.getContext(&quot;2d&quot;),B=t.getContext(&quot;2d&quot;);return B&&(r?B.putImageData(r.getImageData(0,0,A.width,A.height),0,0):B.drawImage(A,0,0)),t}catch(A){}return t},xn.prototype.cloneNode=function(A){if(Qn(A))return document.createTextNode(A.data);if(!A.ownerDocument)return A.cloneNode(!1);var e=A.ownerDocument.defaultView;if(un(A)&&e){var t=this.createElementClone(A),r=e.getComputedStyle(A),B=e.getComputedStyle(A,&quot;:before&quot;),n=e.getComputedStyle(A,&quot;:after&quot;);this.referenceElement===A&&(this.clonedReferenceElement=t),En(t)&&$n(t);for(var s=this.counters.parse(new wB(r)),o=this.resolvePseudoContent(A,t,B,Ln.BEFORE),i=A.firstChild;i;i=i.nextSibling)wn(i)&&(&quot;SCRIPT&quot;===i.tagName||i.hasAttribute(_n)||&quot;function&quot;==typeof this.options.ignoreElements&&this.options.ignoreElements(i))||this.options.copyStyles&&wn(i)&&rn(i)||t.appendChild(this.cloneNode(i));o&&t.insertBefore(o,t.firstChild);var a=this.resolvePseudoContent(A,t,n,Ln.AFTER);return a&&t.appendChild(a),this.counters.pop(s),r&&this.options.copyStyles&&!Hn(A)&&Gn(r,t),0===A.scrollTop&&0===A.scrollLeft||this.scrolledElements.push([t,A.scrollLeft,A.scrollTop]),(dn(A)||fn(A))&&(dn(t)||fn(t))&&(t.value=A.value),t}return A.cloneNode(!1)},xn.prototype.resolvePseudoContent=function(U,A,e,t){var l=this;if(e){var r=e.content,C=A.ownerDocument;if(C&&r&&&quot;none&quot;!==r&&&quot;-moz-alt-content&quot;!==r&&&quot;none&quot;!==e.display){this.counters.parse(new wB(e));var g=new QB(e),E=C.createElement(&quot;html2canvaspseudoelement&quot;);return Gn(e,E),g.content.forEach(function(A){if(A.type===sA.STRING_TOKEN)E.appendChild(C.createTextNode(A.value));else if(A.type===sA.URL_TOKEN){var e=C.createElement(&quot;img&quot;);e.src=A.value,e.style.opacity=&quot;1&quot;,E.appendChild(e)}else if(A.type===sA.FUNCTION){if(&quot;attr&quot;===A.name){var t=A.values.filter(zA);t.length&&E.appendChild(C.createTextNode(U.getAttribute(t[0].value)||&quot;&quot;))}else if(&quot;counter&quot;===A.name){var r=A.values.filter(kA),B=r[0],n=r[1];if(B&&zA(B)){var s=l.counters.getCounterValue(B.value),o=n&&zA(n)?ir.parse(n.value):tr.DECIMAL;E.appendChild(C.createTextNode(yn(s,o,!1)))}}else if(&quot;counters&quot;===A.name){var i=A.values.filter(kA),a=(B=i[0],i[1]);if(n=i[2],B&&zA(B)){var c=l.counters.getCounterValues(B.value),Q=n&&zA(n)?ir.parse(n.value):tr.DECIMAL,w=a&&a.type===sA.STRING_TOKEN?a.value:&quot;&quot;,u=c.map(function(A){return yn(A,Q,!1)}).join(w);E.appendChild(C.createTextNode(u))}}}else if(A.type===sA.IDENT_TOKEN)switch(A.value){case&quot;open-quote&quot;:E.appendChild(C.createTextNode(eB(g.quotes,l.quoteDepth++,!0)));break;case&quot;close-quote&quot;:E.appendChild(C.createTextNode(eB(g.quotes,--l.quoteDepth,!1)))}}),E.className=qn+&quot; &quot;+Zn,A.className+=t===Ln.BEFORE?&quot; &quot;+qn:&quot; &quot;+Zn,E}}},xn);function xn(A,e){if(this.options=e,this.scrolledElements=[],this.referenceElement=A,this.counters=new pn,this.quoteDepth=0,!A.ownerDocument)throw new Error(&quot;Cloned element does not have an owner document&quot;);this.documentElement=this.cloneNode(A.ownerDocument.documentElement)}(On=Ln||(Ln={}))[On.BEFORE=0]=&quot;BEFORE&quot;,On[On.AFTER=1]=&quot;AFTER&quot;;var Vn,zn,Xn=function(A,e){var t=A.createElement(&quot;iframe&quot;);return t.className=&quot;html2canvas-container&quot;,t.style.visibility=&quot;hidden&quot;,t.style.position=&quot;fixed&quot;,t.style.left=&quot;-10000px&quot;,t.style.top=&quot;0px&quot;,t.style.border=&quot;0&quot;,t.width=e.width.toString(),t.height=e.height.toString(),t.scrolling=&quot;no&quot;,t.setAttribute(_n,&quot;true&quot;),A.body.appendChild(t),t},Jn=function(B){return new Promise(function(e,A){var t=B.contentWindow;if(!t)return A(&quot;No window assigned for iframe&quot;);var r=t.document;t.onload=B.onload=r.onreadystatechange=function(){t.onload=B.onload=r.onreadystatechange=null;var A=setInterval(function(){0<r.body.childNodes.length&&&quot;complete&quot;===r.readyState&&(clearInterval(A),e(B))},50)}})},Gn=function(A,e){for(var t=A.length-1;0<=t;t--){var r=A.item(t);&quot;content&quot;!==r&&e.style.setProperty(r,A.getPropertyValue(r))}return e},kn=function(A){var e=&quot;&quot;;return A&&(e+=&quot;<!DOCTYPE &quot;,A.name&&(e+=A.name),A.internalSubset&&(e+=A.internalSubset),A.publicId&&(e+='&quot;'+A.publicId+'&quot;'),A.systemId&&(e+='&quot;'+A.systemId+'&quot;'),e+=&quot;>&quot;),e},Wn=function(A,e,t){A&&A.defaultView&&(e!==A.defaultView.pageXOffset||t!==A.defaultView.pageYOffset)&&A.defaultView.scrollTo(e,t)},Yn=function(A){var e=A[0],t=A[1],r=A[2];e.scrollLeft=t,e.scrollTop=r},qn=&quot;___html2canvas___pseudoelement_before&quot;,Zn=&quot;___html2canvas___pseudoelement_after&quot;,jn='{\n    content: &quot;&quot; !important;\n    display: none !important;\n}',$n=function(A){As(A,&quot;.&quot;+qn+&quot;:before&quot;+jn+&quot;\n         .&quot;+Zn+&quot;:after&quot;+jn)},As=function(A,e){var t=A.ownerDocument;if(t){var r=t.createElement(&quot;style&quot;);r.textContent=e,A.appendChild(r)}};(zn=Vn||(Vn={}))[zn.VECTOR=0]=&quot;VECTOR&quot;,zn[zn.BEZIER_CURVE=1]=&quot;BEZIER_CURVE&quot;;function es(A,t){return A.length===t.length&&A.some(function(A,e){return A===t[e]})}var ts=(rs.prototype.add=function(A,e){return new rs(this.x+A,this.y+e)},rs);function rs(A,e){this.type=Vn.VECTOR,this.x=A,this.y=e}function Bs(A,e,t){return new ts(A.x+(e.x-A.x)*t,A.y+(e.y-A.y)*t)}var ns=(ss.prototype.subdivide=function(A,e){var t=Bs(this.start,this.startControl,A),r=Bs(this.startControl,this.endControl,A),B=Bs(this.endControl,this.end,A),n=Bs(t,r,A),s=Bs(r,B,A),o=Bs(n,s,A);return e?new ss(this.start,t,n,o):new ss(o,s,B,this.end)},ss.prototype.add=function(A,e){return new ss(this.start.add(A,e),this.startControl.add(A,e),this.endControl.add(A,e),this.end.add(A,e))},ss.prototype.reverse=function(){return new ss(this.end,this.endControl,this.startControl,this.start)},ss);function ss(A,e,t,r){this.type=Vn.BEZIER_CURVE,this.start=A,this.startControl=e,this.endControl=t,this.end=r}function os(A){return A.type===Vn.BEZIER_CURVE}var is,as,cs=function(A){var e=A.styles,t=A.bounds,r=jA(e.borderTopLeftRadius,t.width,t.height),B=r[0],n=r[1],s=jA(e.borderTopRightRadius,t.width,t.height),o=s[0],i=s[1],a=jA(e.borderBottomRightRadius,t.width,t.height),c=a[0],Q=a[1],w=jA(e.borderBottomLeftRadius,t.width,t.height),u=w[0],U=w[1],l=[];l.push((B+o)/t.width),l.push((u+c)/t.width),l.push((n+U)/t.height),l.push((i+Q)/t.height);var C=Math.max.apply(Math,l);1<C&&(B/=C,n/=C,o/=C,i/=C,c/=C,Q/=C,u/=C,U/=C);var g=t.width-o,E=t.height-Q,F=t.width-c,h=t.height-U,H=e.borderTopWidth,d=e.borderRightWidth,f=e.borderBottomWidth,p=e.borderLeftWidth,N=ae(e.paddingTop,A.bounds.width),K=ae(e.paddingRight,A.bounds.width),I=ae(e.paddingBottom,A.bounds.width),T=ae(e.paddingLeft,A.bounds.width);this.topLeftBorderBox=0<B||0<n?us(t.left,t.top,B,n,is.TOP_LEFT):new ts(t.left,t.top),this.topRightBorderBox=0<o||0<i?us(t.left+g,t.top,o,i,is.TOP_RIGHT):new ts(t.left+t.width,t.top),this.bottomRightBorderBox=0<c||0<Q?us(t.left+F,t.top+E,c,Q,is.BOTTOM_RIGHT):new ts(t.left+t.width,t.top+t.height),this.bottomLeftBorderBox=0<u||0<U?us(t.left,t.top+h,u,U,is.BOTTOM_LEFT):new ts(t.left,t.top+t.height),this.topLeftPaddingBox=0<B||0<n?us(t.left+p,t.top+H,Math.max(0,B-p),Math.max(0,n-H),is.TOP_LEFT):new ts(t.left+p,t.top+H),this.topRightPaddingBox=0<o||0<i?us(t.left+Math.min(g,t.width+p),t.top+H,g>t.width+p?0:o-p,i-H,is.TOP_RIGHT):new ts(t.left+t.width-d,t.top+H),this.bottomRightPaddingBox=0<c||0<Q?us(t.left+Math.min(F,t.width-p),t.top+Math.min(E,t.height+H),Math.max(0,c-d),Q-f,is.BOTTOM_RIGHT):new ts(t.left+t.width-d,t.top+t.height-f),this.bottomLeftPaddingBox=0<u||0<U?us(t.left+p,t.top+h,Math.max(0,u-p),U-f,is.BOTTOM_LEFT):new ts(t.left+p,t.top+t.height-f),this.topLeftContentBox=0<B||0<n?us(t.left+p+T,t.top+H+N,Math.max(0,B-(p+T)),Math.max(0,n-(H+N)),is.TOP_LEFT):new ts(t.left+p+T,t.top+H+N),this.topRightContentBox=0<o||0<i?us(t.left+Math.min(g,t.width+p+T),t.top+H+N,g>t.width+p+T?0:o-p+T,i-(H+N),is.TOP_RIGHT):new ts(t.left+t.width-(d+K),t.top+H+N),this.bottomRightContentBox=0<c||0<Q?us(t.left+Math.min(F,t.width-(p+T)),t.top+Math.min(E,t.height+H+N),Math.max(0,c-(d+K)),Q-(f+I),is.BOTTOM_RIGHT):new ts(t.left+t.width-(d+K),t.top+t.height-(f+I)),this.bottomLeftContentBox=0<u||0<U?us(t.left+p+T,t.top+h,Math.max(0,u-(p+T)),U-(f+I),is.BOTTOM_LEFT):new ts(t.left+p+T,t.top+t.height-(f+I))};(as=is||(is={}))[as.TOP_LEFT=0]=&quot;TOP_LEFT&quot;,as[as.TOP_RIGHT=1]=&quot;TOP_RIGHT&quot;,as[as.BOTTOM_RIGHT=2]=&quot;BOTTOM_RIGHT&quot;,as[as.BOTTOM_LEFT=3]=&quot;BOTTOM_LEFT&quot;;function Qs(A){return[A.topLeftBorderBox,A.topRightBorderBox,A.bottomRightBorderBox,A.bottomLeftBorderBox]}function ws(A){return[A.topLeftPaddingBox,A.topRightPaddingBox,A.bottomRightPaddingBox,A.bottomLeftPaddingBox]}var us=function(A,e,t,r,B){var n=(Math.sqrt(2)-1)/3*4,s=t*n,o=r*n,i=A+t,a=e+r;switch(B){case is.TOP_LEFT:return new ns(new ts(A,a),new ts(A,a-o),new ts(i-s,e),new ts(i,e));case is.TOP_RIGHT:return new ns(new ts(A,e),new ts(A+s,e),new ts(i,a-o),new ts(i,a));case is.BOTTOM_RIGHT:return new ns(new ts(i,e),new ts(i,e+o),new ts(A+s,a),new ts(A,a));case is.BOTTOM_LEFT:default:return new ns(new ts(i,a),new ts(i-s,a),new ts(A,e+o),new ts(A,e))}},Us=function(A,e,t){this.type=0,this.offsetX=A,this.offsetY=e,this.matrix=t,this.target=6},ls=function(A,e){this.type=1,this.target=e,this.path=A},Cs=function(A){this.element=A,this.inlineLevel=[],this.nonInlineLevel=[],this.negativeZIndex=[],this.zeroOrAutoZIndexOrTransformedOrOpacity=[],this.positiveZIndex=[],this.nonPositionedFloats=[],this.nonPositionedInlineLevel=[]},gs=(Es.prototype.getParentEffects=function(){var A=this.effects.slice(0);if(this.container.styles.overflowX!==sr.VISIBLE){var e=Qs(this.curves),t=ws(this.curves);es(e,t)||A.push(new ls(t,6))}return A},Es);function Es(A,e){if(this.container=A,this.effects=e.slice(0),this.curves=new cs(A),null!==A.styles.transform){var t=A.bounds.left+A.styles.transformOrigin[0].number,r=A.bounds.top+A.styles.transformOrigin[1].number,B=A.styles.transform;this.effects.push(new Us(t,r,B))}if(A.styles.overflowX!==sr.VISIBLE){var n=Qs(this.curves),s=ws(this.curves);es(n,s)?this.effects.push(new ls(n,6)):(this.effects.push(new ls(n,2)),this.effects.push(new ls(s,4)))}}function Fs(A){var e=A.bounds,t=A.styles;return e.add(t.borderLeftWidth,t.borderTopWidth,-(t.borderRightWidth+t.borderLeftWidth),-(t.borderTopWidth+t.borderBottomWidth))}function hs(A){var e=A.styles,t=A.bounds,r=ae(e.paddingLeft,t.width),B=ae(e.paddingRight,t.width),n=ae(e.paddingTop,t.width),s=ae(e.paddingBottom,t.width);return t.add(r+e.borderLeftWidth,n+e.borderTopWidth,-(e.borderRightWidth+e.borderLeftWidth+r+B),-(e.borderTopWidth+e.borderBottomWidth+n+s))}function Hs(A,e,t){var r=function(A,e){return 0===A?e.bounds:2===A?hs(e):Fs(e)}(Ts(A.styles.backgroundOrigin,e),A),B=function(A,e){return A===Ee.BORDER_BOX?e.bounds:A===Ee.CONTENT_BOX?hs(e):Fs(e)}(Ts(A.styles.backgroundClip,e),A),n=Is(Ts(A.styles.backgroundSize,e),t,r),s=n[0],o=n[1],i=jA(Ts(A.styles.backgroundPosition,e),r.width-s,r.height-o);return[ms(Ts(A.styles.backgroundRepeat,e),i,n,r,B),Math.round(r.left+i[0]),Math.round(r.top+i[1]),s,o]}function ds(A){return zA(A)&&A.value===Ut.AUTO}function fs(A){return&quot;number&quot;==typeof A}var ps=function(c,Q,w,u){c.container.elements.forEach(function(A){var e=AB(A.flags,4),t=AB(A.flags,2),r=new gs(A,c.getParentEffects());AB(A.styles.display,2048)&&u.push(r);var B=AB(A.flags,8)?[]:u;if(e||t){var n=e||A.styles.isPositioned()?w:Q,s=new Cs(r);if(A.styles.isPositioned()||A.styles.opacity<1||A.styles.isTransformed()){var o=A.styles.zIndex.order;if(o<0){var i=0;n.negativeZIndex.some(function(A,e){return o>A.element.container.styles.zIndex.order&&(i=e,!0)}),n.negativeZIndex.splice(i,0,s)}else if(0<o){var a=0;n.positiveZIndex.some(function(A,e){return o>A.element.container.styles.zIndex.order&&(a=e+1,!0)}),n.positiveZIndex.splice(a,0,s)}else n.zeroOrAutoZIndexOrTransformedOrOpacity.push(s)}else A.styles.isFloating()?n.nonPositionedFloats.push(s):n.nonPositionedInlineLevel.push(s);ps(r,s,e?s:w,B)}else A.styles.isInlineLevel()?Q.inlineLevel.push(r):Q.nonInlineLevel.push(r),ps(r,Q,w,B);AB(A.flags,8)&&Ns(A,B)})},Ns=function(A,e){for(var t=A instanceof bB?A.start:1,r=A instanceof bB&&A.reversed,B=0;B<e.length;B++){var n=e[B];n.container instanceof vB&&&quot;number&quot;==typeof n.container.value&&0!==n.container.value&&(t=n.container.value),n.listValue=yn(t,n.container.styles.listStyleType,!0),t+=r?-1:1}},Ks=function(A,e,t,r){var B=[];return os(A)?B.push(A.subdivide(.5,!1)):B.push(A),os(t)?B.push(t.subdivide(.5,!0)):B.push(t),os(r)?B.push(r.subdivide(.5,!0).reverse()):B.push(r),os(e)?B.push(e.subdivide(.5,!1).reverse()):B.push(e),B},Is=function(A,e,t){var r=e[0],B=e[1],n=e[2],s=A[0],o=A[1];if(qA(s)&&o&&qA(o))return[ae(s,t.width),ae(o,t.height)];var i=fs(n);if(zA(s)&&(s.value===Ut.CONTAIN||s.value===Ut.COVER))return fs(n)?t.width/t.height<n!=(s.value===Ut.COVER)?[t.width,t.width/n]:[t.height*n,t.height]:[t.width,t.height];var a=fs(r),c=fs(B),Q=a||c;if(ds(s)&&(!o||ds(o)))return a&&c?[r,B]:i||Q?Q&&i?[a?r:B*n,c?B:r/n]:[a?r:t.width,c?B:t.height]:[t.width,t.height];if(i){var w=0,u=0;return qA(s)?w=ae(s,t.width):qA(o)&&(u=ae(o,t.height)),ds(s)?w=u*n:o&&!ds(o)||(u=w/n),[w,u]}var U=null,l=null;if(qA(s)?U=ae(s,t.width):o&&qA(o)&&(l=ae(o,t.height)),null===U||o&&!ds(o)||(l=a&&c?U/r*B:t.height),null!==l&&ds(s)&&(U=a&&c?l/B*r:t.width),null!==U&&null!==l)return[U,l];throw new Error(&quot;Unable to calculate background-size for element&quot;)},Ts=function(A,e){var t=A[e];return void 0===t?A[0]:t},ms=function(A,e,t,r,B){var n=e[0],s=e[1],o=t[0],i=t[1];switch(A){case ot.REPEAT_X:return[new ts(Math.round(r.left),Math.round(r.top+s)),new ts(Math.round(r.left+r.width),Math.round(r.top+s)),new ts(Math.round(r.left+r.width),Math.round(i+r.top+s)),new ts(Math.round(r.left),Math.round(i+r.top+s))];case ot.REPEAT_Y:return[new ts(Math.round(r.left+n),Math.round(r.top)),new ts(Math.round(r.left+n+o),Math.round(r.top)),new ts(Math.round(r.left+n+o),Math.round(r.height+r.top)),new ts(Math.round(r.left+n),Math.round(r.height+r.top))];case ot.NO_REPEAT:return[new ts(Math.round(r.left+n),Math.round(r.top+s)),new ts(Math.round(r.left+n+o),Math.round(r.top+s)),new ts(Math.round(r.left+n+o),Math.round(r.top+s+i)),new ts(Math.round(r.left+n),Math.round(r.top+s+i))];default:return[new ts(Math.round(B.left),Math.round(B.top)),new ts(Math.round(B.left+B.width),Math.round(B.top)),new ts(Math.round(B.left+B.width),Math.round(B.height+B.top)),new ts(Math.round(B.left),Math.round(B.height+B.top))]}},Rs=&quot;Hidden Text&quot;,Ls=(Os.prototype.parseMetrics=function(A,e){var t=this._document.createElement(&quot;div&quot;),r=this._document.createElement(&quot;img&quot;),B=this._document.createElement(&quot;span&quot;),n=this._document.body;t.style.visibility=&quot;hidden&quot;,t.style.fontFamily=A,t.style.fontSize=e,t.style.margin=&quot;0&quot;,t.style.padding=&quot;0&quot;,n.appendChild(t),r.src=&quot;data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7&quot;,r.width=1,r.height=1,r.style.margin=&quot;0&quot;,r.style.padding=&quot;0&quot;,r.style.verticalAlign=&quot;baseline&quot;,B.style.fontFamily=A,B.style.fontSize=e,B.style.margin=&quot;0&quot;,B.style.padding=&quot;0&quot;,B.appendChild(this._document.createTextNode(Rs)),t.appendChild(B),t.appendChild(r);var s=r.offsetTop-B.offsetTop+2;t.removeChild(B),t.appendChild(this._document.createTextNode(Rs)),t.style.lineHeight=&quot;normal&quot;,r.style.verticalAlign=&quot;super&quot;;var o=r.offsetTop-t.offsetTop+2;return n.removeChild(t),{baseline:s,middle:o}},Os.prototype.getMetrics=function(A,e){var t=A+&quot; &quot;+e;return void 0===this._data[t]&&(this._data[t]=this.parseMetrics(A,e)),this._data[t]},Os);function Os(A){this._data={},this._document=A}var vs=(Ds.prototype.applyEffects=function(A,e){for(var t=this;this._activeEffects.length;)this.popEffect();A.filter(function(A){return AB(A.target,e)}).forEach(function(A){return t.applyEffect(A)})},Ds.prototype.applyEffect=function(A){this.ctx.save(),function(A){return 0===A.type}(A)&&(this.ctx.translate(A.offsetX,A.offsetY),this.ctx.transform(A.matrix[0],A.matrix[1],A.matrix[2],A.matrix[3],A.matrix[4],A.matrix[5]),this.ctx.translate(-A.offsetX,-A.offsetY)),function(A){return 1===A.type}(A)&&(this.path(A.path),this.ctx.clip()),this._activeEffects.push(A)},Ds.prototype.popEffect=function(){this._activeEffects.pop(),this.ctx.restore()},Ds.prototype.renderStack=function(t){return B(this,void 0,void 0,function(){var e;return b(this,function(A){switch(A.label){case 0:return(e=t.element.container.styles).isVisible()?(this.ctx.globalAlpha=e.opacity,[4,this.renderStackContent(t)]):[3,2];case 1:A.sent(),A.label=2;case 2:return[2]}})})},Ds.prototype.renderNode=function(e){return B(this,void 0,void 0,function(){return b(this,function(A){switch(A.label){case 0:return e.container.styles.isVisible()?[4,this.renderNodeBackgroundAndBorders(e)]:[3,3];case 1:return A.sent(),[4,this.renderNodeContent(e)];case 2:A.sent(),A.label=3;case 3:return[2]}})})},Ds.prototype.renderTextWithLetterSpacing=function(t,A){var r=this;0===A?this.ctx.fillText(t.text,t.bounds.left,t.bounds.top+t.bounds.height):c(t.text).map(function(A){return l(A)}).reduce(function(A,e){return r.ctx.fillText(e,A,t.bounds.top+t.bounds.height),A+r.ctx.measureText(e).width},t.bounds.left)},Ds.prototype.createFontStyle=function(A){var e=A.fontVariant.filter(function(A){return&quot;normal&quot;===A||&quot;small-caps&quot;===A}).join(&quot;&quot;),t=A.fontFamily.join(&quot;, &quot;),r=xA(A.fontSize)?&quot;&quot;+A.fontSize.number+A.fontSize.unit:A.fontSize.number+&quot;px&quot;;return[[A.fontStyle,e,A.fontWeight,r,t].join(&quot; &quot;),t,r]},Ds.prototype.renderTextNode=function(r,o){return B(this,void 0,void 0,function(){var e,t,B,n,s=this;return b(this,function(A){return e=this.createFontStyle(o),t=e[0],B=e[1],n=e[2],this.ctx.font=t,r.textBounds.forEach(function(r){s.ctx.fillStyle=te(o.color),s.renderTextWithLetterSpacing(r,o.letterSpacing);var A=o.textShadow;A.length&&r.text.trim().length&&(A.slice(0).reverse().forEach(function(A){s.ctx.shadowColor=te(A.color),s.ctx.shadowOffsetX=A.offsetX.number*s.options.scale,s.ctx.shadowOffsetY=A.offsetY.number*s.options.scale,s.ctx.shadowBlur=A.blur.number,s.ctx.fillText(r.text,r.bounds.left,r.bounds.top+r.bounds.height)}),s.ctx.shadowColor=&quot;&quot;,s.ctx.shadowOffsetX=0,s.ctx.shadowOffsetY=0,s.ctx.shadowBlur=0),o.textDecorationLine.length&&(s.ctx.fillStyle=te(o.textDecorationColor||o.color),o.textDecorationLine.forEach(function(A){switch(A){case 1:var e=s.fontMetrics.getMetrics(B,n).baseline;s.ctx.fillRect(r.bounds.left,Math.round(r.bounds.top+e),r.bounds.width,1);break;case 2:s.ctx.fillRect(r.bounds.left,Math.round(r.bounds.top),r.bounds.width,1);break;case 3:var t=s.fontMetrics.getMetrics(B,n).middle;s.ctx.fillRect(r.bounds.left,Math.ceil(r.bounds.top+t),r.bounds.width,1)}}))}),[2]})})},Ds.prototype.renderReplacedElement=function(A,e,t){if(t&&0<A.intrinsicWidth&&0<A.intrinsicHeight){var r=hs(A),B=ws(e);this.path(B),this.ctx.save(),this.ctx.clip(),this.ctx.drawImage(t,0,0,A.intrinsicWidth,A.intrinsicHeight,r.left,r.top,r.width,r.height),this.ctx.restore()}},Ds.prototype.renderNodeContent=function(l){return B(this,void 0,void 0,function(){var e,t,r,B,n,s,o,i,a,c,Q,w,u,U;return b(this,function(A){switch(A.label){case 0:this.applyEffects(l.effects,4),e=l.container,t=l.curves,r=e.styles,B=0,n=e.textNodes,A.label=1;case 1:return B<n.length?(s=n[B],[4,this.renderTextNode(s,r)]):[3,4];case 2:A.sent(),A.label=3;case 3:return B++,[3,1];case 4:if(!(e instanceof pB))return[3,8];A.label=5;case 5:return A.trys.push([5,7,,8]),[4,this.options.cache.match(e.src)];case 6:return w=A.sent(),this.renderReplacedElement(e,t,w),[3,8];case 7:return A.sent(),De.getInstance(this.options.id).error(&quot;Error loading image &quot;+e.src),[3,8];case 8:if(e instanceof IB&&this.renderReplacedElement(e,t,e.canvas),!(e instanceof RB))return[3,12];A.label=9;case 9:return A.trys.push([9,11,,12]),[4,this.options.cache.match(e.svg)];case 10:return w=A.sent(),this.renderReplacedElement(e,t,w),[3,12];case 11:return A.sent(),De.getInstance(this.options.id).error(&quot;Error loading svg &quot;+e.svg.substring(0,255)),[3,12];case 12:return e instanceof en&&e.tree?[4,new Ds({id:this.options.id,scale:this.options.scale,backgroundColor:e.backgroundColor,x:0,y:0,scrollX:0,scrollY:0,width:e.width,height:e.height,cache:this.options.cache,windowWidth:e.width,windowHeight:e.height}).render(e.tree)]:[3,14];case 13:o=A.sent(),this.ctx.drawImage(o,0,0,e.width,e.width,e.bounds.left,e.bounds.top,e.bounds.width,e.bounds.height),A.label=14;case 14:if(e instanceof JB&&(i=Math.min(e.bounds.width,e.bounds.height),e.type===xB?e.checked&&(this.ctx.save(),this.path([new ts(e.bounds.left+.39363*i,e.bounds.top+.79*i),new ts(e.bounds.left+.16*i,e.bounds.top+.5549*i),new ts(e.bounds.left+.27347*i,e.bounds.top+.44071*i),new ts(e.bounds.left+.39694*i,e.bounds.top+.5649*i),new ts(e.bounds.left+.72983*i,e.bounds.top+.23*i),new ts(e.bounds.left+.84*i,e.bounds.top+.34085*i),new ts(e.bounds.left+.39363*i,e.bounds.top+.79*i)]),this.ctx.fillStyle=te(XB),this.ctx.fill(),this.ctx.restore()):e.type===VB&&e.checked&&(this.ctx.save(),this.ctx.beginPath(),this.ctx.arc(e.bounds.left+i/2,e.bounds.top+i/2,i/4,0,2*Math.PI,!0),this.ctx.fillStyle=te(XB),this.ctx.fill(),this.ctx.restore())),Ss(e)&&e.value.length){switch(this.ctx.font=this.createFontStyle(r)[0],this.ctx.fillStyle=te(r.color),this.ctx.textBaseline=&quot;middle&quot;,this.ctx.textAlign=Ms(e.styles.textAlign),U=hs(e),a=0,e.styles.textAlign){case Cr.CENTER:a+=U.width/2;break;case Cr.RIGHT:a+=U.width}c=U.add(a,0,0,-U.height/2+1),this.ctx.save(),this.path([new ts(U.left,U.top),new ts(U.left+U.width,U.top),new ts(U.left+U.width,U.top+U.height),new ts(U.left,U.top+U.height)]),this.ctx.clip(),this.renderTextWithLetterSpacing(new lB(e.value,c),r.letterSpacing),this.ctx.restore(),this.ctx.textBaseline=&quot;bottom&quot;,this.ctx.textAlign=&quot;left&quot;}if(!AB(e.styles.display,2048))return[3,20];if(null===e.styles.listStyleImage)return[3,19];if((Q=e.styles.listStyleImage).type!==xe.URL)return[3,18];w=void 0,u=Q.url,A.label=15;case 15:return A.trys.push([15,17,,18]),[4,this.options.cache.match(u)];case 16:return w=A.sent(),this.ctx.drawImage(w,e.bounds.left-(w.width+10),e.bounds.top),[3,18];case 17:return A.sent(),De.getInstance(this.options.id).error(&quot;Error loading list-style-image &quot;+u),[3,18];case 18:return[3,20];case 19:l.listValue&&e.styles.listStyleType!==tr.NONE&&(this.ctx.font=this.createFontStyle(r)[0],this.ctx.fillStyle=te(r.color),this.ctx.textBaseline=&quot;middle&quot;,this.ctx.textAlign=&quot;right&quot;,U=new I(e.bounds.left,e.bounds.top+ae(e.styles.paddingTop,e.bounds.width),e.bounds.width,function(A,e){return zA(A)&&&quot;normal&quot;===A.value?1.2*e:A.type===sA.NUMBER_TOKEN?e*A.number:qA(A)?ae(A,e):e}(r.lineHeight,r.fontSize.number)/2+1),this.renderTextWithLetterSpacing(new lB(l.listValue,U),r.letterSpacing),this.ctx.textBaseline=&quot;bottom&quot;,this.ctx.textAlign=&quot;left&quot;),A.label=20;case 20:return[2]}})})},Ds.prototype.renderStackContent=function(C){return B(this,void 0,void 0,function(){var e,t,r,B,n,s,o,i,a,c,Q,w,u,U,l;return b(this,function(A){switch(A.label){case 0:return[4,this.renderNodeBackgroundAndBorders(C.element)];case 1:A.sent(),e=0,t=C.negativeZIndex,A.label=2;case 2:return e<t.length?(l=t[e],[4,this.renderStack(l)]):[3,5];case 3:A.sent(),A.label=4;case 4:return e++,[3,2];case 5:return[4,this.renderNodeContent(C.element)];case 6:A.sent(),r=0,B=C.nonInlineLevel,A.label=7;case 7:return r<B.length?(l=B[r],[4,this.renderNode(l)]):[3,10];case 8:A.sent(),A.label=9;case 9:return r++,[3,7];case 10:n=0,s=C.nonPositionedFloats,A.label=11;case 11:return n<s.length?(l=s[n],[4,this.renderStack(l)]):[3,14];case 12:A.sent(),A.label=13;case 13:return n++,[3,11];case 14:o=0,i=C.nonPositionedInlineLevel,A.label=15;case 15:return o<i.length?(l=i[o],[4,this.renderStack(l)]):[3,18];case 16:A.sent(),A.label=17;case 17:return o++,[3,15];case 18:a=0,c=C.inlineLevel,A.label=19;case 19:return a<c.length?(l=c[a],[4,this.renderNode(l)]):[3,22];case 20:A.sent(),A.label=21;case 21:return a++,[3,19];case 22:Q=0,w=C.zeroOrAutoZIndexOrTransformedOrOpacity,A.label=23;case 23:return Q<w.length?(l=w[Q],[4,this.renderStack(l)]):[3,26];case 24:A.sent(),A.label=25;case 25:return Q++,[3,23];case 26:u=0,U=C.positiveZIndex,A.label=27;case 27:return u<U.length?(l=U[u],[4,this.renderStack(l)]):[3,30];case 28:A.sent(),A.label=29;case 29:return u++,[3,27];case 30:return[2]}})})},Ds.prototype.mask=function(A){this.ctx.beginPath(),this.ctx.moveTo(0,0),this.ctx.lineTo(this.canvas.width,0),this.ctx.lineTo(this.canvas.width,this.canvas.height),this.ctx.lineTo(0,this.canvas.height),this.ctx.lineTo(0,0),this.formatPath(A.slice(0).reverse()),this.ctx.closePath()},Ds.prototype.path=function(A){this.ctx.beginPath(),this.formatPath(A),this.ctx.closePath()},Ds.prototype.formatPath=function(A){var r=this;A.forEach(function(A,e){var t=os(A)?A.start:A;0===e?r.ctx.moveTo(t.x,t.y):r.ctx.lineTo(t.x,t.y),os(A)&&r.ctx.bezierCurveTo(A.startControl.x,A.startControl.y,A.endControl.x,A.endControl.y,A.end.x,A.end.y)})},Ds.prototype.renderRepeat=function(A,e,t,r){this.path(A),this.ctx.fillStyle=e,this.ctx.translate(t,r),this.ctx.fill(),this.ctx.translate(-t,-r)},Ds.prototype.resizeImage=function(A,e,t){if(A.width===e&&A.height===t)return A;var r=this.canvas.ownerDocument.createElement(&quot;canvas&quot;);return r.width=e,r.height=t,r.getContext(&quot;2d&quot;).drawImage(A,0,0,A.width,A.height,0,0,e,t),r},Ds.prototype.renderBackgroundImage=function(S){return B(this,void 0,void 0,function(){var v,e,D,t,r,B;return b(this,function(A){switch(A.label){case 0:v=S.styles.backgroundImage.length-1,e=function(e){var t,r,B,n,s,o,i,a,c,Q,w,u,U,l,C,g,E,F,h,H,d,f,p,N,K,I,T,m,R,L,O;return b(this,function(A){switch(A.label){case 0:if(e.type!==xe.URL)return[3,5];t=void 0,r=e.url,A.label=1;case 1:return A.trys.push([1,3,,4]),[4,D.options.cache.match(r)];case 2:return t=A.sent(),[3,4];case 3:return A.sent(),De.getInstance(D.options.id).error(&quot;Error loading background-image &quot;+r),[3,4];case 4:return t&&(B=Hs(S,v,[t.width,t.height,t.width/t.height]),g=B[0],f=B[1],p=B[2],h=B[3],H=B[4],l=D.ctx.createPattern(D.resizeImage(t,h,H),&quot;repeat&quot;),D.renderRepeat(g,l,f,p)),[3,6];case 5:!function(A){return A.type===xe.LINEAR_GRADIENT}(e)?function(A){return A.type===xe.RADIAL_GRADIENT}(e)&&(C=Hs(S,v,[null,null,null]),g=C[0],E=C[1],F=C[2],h=C[3],H=C[4],d=0===e.position.length?[oe]:e.position,f=ae(d[0],h),p=ae(d[d.length-1],H),N=function(A,e,t,r,B){var n=0,s=0;switch(A.size){case nt.CLOSEST_SIDE:A.shape===rt.CIRCLE?n=s=Math.min(Math.abs(e),Math.abs(e-r),Math.abs(t),Math.abs(t-B)):A.shape===rt.ELLIPSE&&(n=Math.min(Math.abs(e),Math.abs(e-r)),s=Math.min(Math.abs(t),Math.abs(t-B)));break;case nt.CLOSEST_CORNER:if(A.shape===rt.CIRCLE)n=s=Math.min(Ne(e,t),Ne(e,t-B),Ne(e-r,t),Ne(e-r,t-B));else if(A.shape===rt.ELLIPSE){var o=Math.min(Math.abs(t),Math.abs(t-B))/Math.min(Math.abs(e),Math.abs(e-r)),i=Ke(r,B,e,t,!0),a=i[0],c=i[1];s=o*(n=Ne(a-e,(c-t)/o))}break;case nt.FARTHEST_SIDE:A.shape===rt.CIRCLE?n=s=Math.max(Math.abs(e),Math.abs(e-r),Math.abs(t),Math.abs(t-B)):A.shape===rt.ELLIPSE&&(n=Math.max(Math.abs(e),Math.abs(e-r)),s=Math.max(Math.abs(t),Math.abs(t-B)));break;case nt.FARTHEST_CORNER:if(A.shape===rt.CIRCLE)n=s=Math.max(Ne(e,t),Ne(e,t-B),Ne(e-r,t),Ne(e-r,t-B));else if(A.shape===rt.ELLIPSE){o=Math.max(Math.abs(t),Math.abs(t-B))/Math.max(Math.abs(e),Math.abs(e-r));var Q=Ke(r,B,e,t,!1);a=Q[0],c=Q[1],s=o*(n=Ne(a-e,(c-t)/o))}}return Array.isArray(A.size)&&(n=ae(A.size[0],r),s=2===A.size.length?ae(A.size[1],B):n),[n,s]}(e,f,p,h,H),K=N[0],I=N[1],0<K&&0<K&&(T=D.ctx.createRadialGradient(E+f,F+p,0,E+f,F+p,K),fe(e.stops,2*K).forEach(function(A){return T.addColorStop(A.stop,te(A.color))}),D.path(g),D.ctx.fillStyle=T,K!==I?(m=S.bounds.left+.5*S.bounds.width,R=S.bounds.top+.5*S.bounds.height,O=1/(L=I/K),D.ctx.save(),D.ctx.translate(m,R),D.ctx.transform(1,0,0,L,0,0),D.ctx.translate(-m,-R),D.ctx.fillRect(E,O*(F-R)+R,h,H*O),D.ctx.restore()):D.ctx.fill())):(n=Hs(S,v,[null,null,null]),g=n[0],f=n[1],p=n[2],h=n[3],H=n[4],s=pe(e.angle,h,H),o=s[0],i=s[1],a=s[2],c=s[3],Q=s[4],(w=document.createElement(&quot;canvas&quot;)).width=h,w.height=H,u=w.getContext(&quot;2d&quot;),U=u.createLinearGradient(i,c,a,Q),fe(e.stops,o).forEach(function(A){return U.addColorStop(A.stop,te(A.color))}),u.fillStyle=U,u.fillRect(0,0,h,H),l=D.ctx.createPattern(w,&quot;repeat&quot;),D.renderRepeat(g,l,f,p)),A.label=6;case 6:return v--,[2]}})},D=this,t=0,r=S.styles.backgroundImage.slice(0).reverse(),A.label=1;case 1:return t<r.length?(B=r[t],[5,e(B)]):[3,4];case 2:A.sent(),A.label=3;case 3:return t++,[3,1];case 4:return[2]}})})},Ds.prototype.renderBorder=function(e,t,r){return B(this,void 0,void 0,function(){return b(this,function(A){return this.path(function(A,e){switch(e){case 0:return Ks(A.topLeftBorderBox,A.topLeftPaddingBox,A.topRightBorderBox,A.topRightPaddingBox);case 1:return Ks(A.topRightBorderBox,A.topRightPaddingBox,A.bottomRightBorderBox,A.bottomRightPaddingBox);case 2:return Ks(A.bottomRightBorderBox,A.bottomRightPaddingBox,A.bottomLeftBorderBox,A.bottomLeftPaddingBox);case 3:default:return Ks(A.bottomLeftBorderBox,A.bottomLeftPaddingBox,A.topLeftBorderBox,A.topLeftPaddingBox)}}(r,t)),this.ctx.fillStyle=te(e),this.ctx.fill(),[2]})})},Ds.prototype.renderNodeBackgroundAndBorders=function(c){return B(this,void 0,void 0,function(){var e,t,r,B,n,s,o,i,a=this;return b(this,function(A){switch(A.label){case 0:return this.applyEffects(c.effects,2),e=c.container.styles,t=!ee(e.backgroundColor)||e.backgroundImage.length,r=[{style:e.borderTopStyle,color:e.borderTopColor},{style:e.borderRightStyle,color:e.borderRightColor},{style:e.borderBottomStyle,color:e.borderBottomColor},{style:e.borderLeftStyle,color:e.borderLeftColor}],B=bs(Ts(e.backgroundClip,0),c.curves),t||e.boxShadow.length?(this.ctx.save(),this.path(B),this.ctx.clip(),ee(e.backgroundColor)||(this.ctx.fillStyle=te(e.backgroundColor),this.ctx.fill()),[4,this.renderBackgroundImage(c.container)]):[3,2];case 1:A.sent(),this.ctx.restore(),e.boxShadow.slice(0).reverse().forEach(function(A){a.ctx.save();var e=Qs(c.curves),t=A.inset?0:1e4,r=function(A,t,r,B,n){return A.map(function(A,e){switch(e){case 0:return A.add(t,r);case 1:return A.add(t+B,r);case 2:return A.add(t+B,r+n);case 3:return A.add(t,r+n)}return A})}(e,-t+(A.inset?1:-1)*A.spread.number,(A.inset?1:-1)*A.spread.number,A.spread.number*(A.inset?-2:2),A.spread.number*(A.inset?-2:2));A.inset?(a.path(e),a.ctx.clip(),a.mask(r)):(a.mask(e),a.ctx.clip(),a.path(r)),a.ctx.shadowOffsetX=A.offsetX.number+t,a.ctx.shadowOffsetY=A.offsetY.number,a.ctx.shadowColor=te(A.color),a.ctx.shadowBlur=A.blur.number,a.ctx.fillStyle=A.inset?te(A.color):&quot;rgba(0,0,0,1)&quot;,a.ctx.fill(),a.ctx.restore()}),A.label=2;case 2:s=n=0,o=r,A.label=3;case 3:return s<o.length?(i=o[s]).style===ht.NONE||ee(i.color)?[3,5]:[4,this.renderBorder(i.color,n++,c.curves)]:[3,6];case 4:A.sent(),A.label=5;case 5:return s++,[3,3];case 6:return[2]}})})},Ds.prototype.render=function(t){return B(this,void 0,void 0,function(){var e;return b(this,function(A){switch(A.label){case 0:return this.options.backgroundColor&&(this.ctx.fillStyle=te(this.options.backgroundColor),this.ctx.fillRect(this.options.x-this.options.scrollX,this.options.y-this.options.scrollY,this.options.width,this.options.height)),e=function(A){var e=new gs(A,[]),t=new Cs(e),r=[];return ps(e,t,t,r),Ns(e.container,r),t}(t),[4,this.renderStack(e)];case 1:return A.sent(),this.applyEffects([],2),[2,this.canvas]}})})},Ds);function Ds(A){this._activeEffects=[],this.canvas=A.canvas?A.canvas:document.createElement(&quot;canvas&quot;),this.ctx=this.canvas.getContext(&quot;2d&quot;),this.options=A,this.canvas.width=Math.floor(A.width*A.scale),this.canvas.height=Math.floor(A.height*A.scale),this.canvas.style.width=A.width+&quot;px&quot;,this.canvas.style.height=A.height+&quot;px&quot;,this.fontMetrics=new Ls(document),this.ctx.scale(this.options.scale,this.options.scale),this.ctx.translate(-A.x+A.scrollX,-A.y+A.scrollY),this.ctx.textBaseline=&quot;bottom&quot;,this._activeEffects=[],De.getInstance(A.id).debug(&quot;Canvas renderer initialized (&quot;+A.width+&quot;x&quot;+A.height+&quot; at &quot;+A.x+&quot;,&quot;+A.y+&quot;) with scale &quot;+A.scale)}var Ss=function(A){return A instanceof ZB||(A instanceof WB||A instanceof JB&&A.type!==VB&&A.type!==xB)},bs=function(A,e){switch(A){case Ee.BORDER_BOX:return Qs(e);case Ee.CONTENT_BOX:return function(A){return[A.topLeftContentBox,A.topRightContentBox,A.bottomRightContentBox,A.bottomLeftContentBox]}(e);case Ee.PADDING_BOX:default:return ws(e)}},Ms=function(A){switch(A){case Cr.CENTER:return&quot;center&quot;;case Cr.RIGHT:return&quot;right&quot;;case Cr.LEFT:default:return&quot;left&quot;}},ys=(_s.prototype.render=function(r){return B(this,void 0,void 0,function(){var e,t;return b(this,function(A){switch(A.label){case 0:return e=Le(Math.max(this.options.windowWidth,this.options.width)*this.options.scale,Math.max(this.options.windowHeight,this.options.height)*this.options.scale,this.options.scrollX*this.options.scale,this.options.scrollY*this.options.scale,r),[4,xs(e)];case 1:return t=A.sent(),this.options.backgroundColor&&(this.ctx.fillStyle=te(this.options.backgroundColor),this.ctx.fillRect(0,0,this.options.width*this.options.scale,this.options.height*this.options.scale)),this.ctx.drawImage(t,-this.options.x*this.options.scale,-this.options.y*this.options.scale),[2,this.canvas]}})})},_s);function _s(A){this.canvas=A.canvas?A.canvas:document.createElement(&quot;canvas&quot;),this.ctx=this.canvas.getContext(&quot;2d&quot;),this.options=A,this.canvas.width=Math.floor(A.width*A.scale),this.canvas.height=Math.floor(A.height*A.scale),this.canvas.style.width=A.width+&quot;px&quot;,this.canvas.style.height=A.height+&quot;px&quot;,this.ctx.scale(this.options.scale,this.options.scale),this.ctx.translate(-A.x+A.scrollX,-A.y+A.scrollY),De.getInstance(A.id).debug(&quot;EXPERIMENTAL ForeignObject renderer initialized (&quot;+A.width+&quot;x&quot;+A.height+&quot; at &quot;+A.x+&quot;,&quot;+A.y+&quot;) with scale &quot;+A.scale)}function Ps(A){return we(_A.create(A).parseComponentValue())}var xs=function(r){return new Promise(function(A,e){var t=new Image;t.onload=function(){A(t)},t.onerror=e,t.src=&quot;data:image/svg+xml;charset=utf-8,&quot;+encodeURIComponent((new XMLSerializer).serializeToString(r))})};be.setContext(window);var Vs=function(p,N){return B(void 0,void 0,void 0,function(){var e,t,r,B,n,s,o,i,a,c,Q,w,u,U,l,C,g,E,F,h,H,d,f;return b(this,function(A){switch(A.label){case 0:if(!(e=p.ownerDocument))throw new Error(&quot;Element is not attached to a Document&quot;);if(!(t=e.defaultView))throw new Error(&quot;Document is not attached to a Window&quot;);return r=(Math.round(1e3*Math.random())+Date.now()).toString(16),B=En(p)||function(A){return&quot;HTML&quot;===A.tagName}(p)?function(A){var e=A.body,t=A.documentElement;if(!e||!t)throw new Error(&quot;Unable to get document size&quot;);var r=Math.max(Math.max(e.scrollWidth,t.scrollWidth),Math.max(e.offsetWidth,t.offsetWidth),Math.max(e.clientWidth,t.clientWidth)),B=Math.max(Math.max(e.scrollHeight,t.scrollHeight),Math.max(e.offsetHeight,t.offsetHeight),Math.max(e.clientHeight,t.clientHeight));return new I(0,0,r,B)}(e):T(p),n=B.width,s=B.height,o=B.left,i=B.top,a=K({},{allowTaint:!1,imageTimeout:15e3,proxy:void 0,useCORS:!1},N),c={backgroundColor:&quot;#ffffff&quot;,cache:N.cache?N.cache:be.create(r,a),logging:!0,removeContainer:!0,foreignObjectRendering:!1,scale:t.devicePixelRatio||1,windowWidth:t.innerWidth,windowHeight:t.innerHeight,scrollX:t.pageXOffset,scrollY:t.pageYOffset,x:o,y:i,width:Math.ceil(n),height:Math.ceil(s),id:r},Q=K({},c,a,N),w=new I(Q.scrollX,Q.scrollY,Q.windowWidth,Q.windowHeight),De.create(r),De.getInstance(r).debug(&quot;Starting document clone&quot;),u=new Pn(p,{id:r,onclone:Q.onclone,ignoreElements:Q.ignoreElements,inlineImages:Q.foreignObjectRendering,copyStyles:Q.foreignObjectRendering}),(U=u.clonedReferenceElement)?[4,u.toIFrame(e,w)]:[2,Promise.reject(&quot;Unable to find element in cloned iframe&quot;)];case 1:return l=A.sent(),C=e.documentElement?Ps(getComputedStyle(e.documentElement).backgroundColor):He.TRANSPARENT,g=e.body?Ps(getComputedStyle(e.body).backgroundColor):He.TRANSPARENT,E=N.backgroundColor,F=&quot;string&quot;==typeof E?Ps(E):4294967295,h=p===e.documentElement?ee(C)?ee(g)?F:g:C:F,H={id:r,cache:Q.cache,backgroundColor:h,scale:Q.scale,x:Q.x,y:Q.y,scrollX:Q.scrollX,scrollY:Q.scrollY,width:Q.width,height:Q.height,windowWidth:Q.windowWidth,windowHeight:Q.windowHeight},Q.foreignObjectRendering?(De.getInstance(r).debug(&quot;Document cloned, using foreign object rendering&quot;),[4,new ys(H).render(U)]):[3,3];case 2:return d=A.sent(),[3,5];case 3:return De.getInstance(r).debug(&quot;Document cloned, using computed rendering&quot;),be.attachInstance(Q.cache),De.getInstance(r).debug(&quot;Starting DOM parsing&quot;),f=on(U),be.detachInstance(),h===f.styles.backgroundColor&&(f.styles.backgroundColor=He.TRANSPARENT),De.getInstance(r).debug(&quot;Starting renderer&quot;),[4,new vs(H).render(f)];case 4:d=A.sent(),A.label=5;case 5:return!0===Q.removeContainer&&(zs(l)||De.getInstance(r).error(&quot;Cannot detach cloned iframe as it is not in the DOM anymore&quot;)),De.getInstance(r).debug(&quot;Finished rendering&quot;),De.destroy(r),be.destroy(r),[2,d]}})})},zs=function(A){return!!A.parentNode&&(A.parentNode.removeChild(A),!0)};return function(A,e){return void 0===e&&(e={}),Vs(A,e)}});
</script>
<script>
  var init = (function () {
  'use strict';

  const BASEMAPS = {
    DarkMatter: carto.basemaps.darkmatter,
    Voyager: carto.basemaps.voyager,
    Positron: carto.basemaps.positron
  };

  const attributionControl = new mapboxgl.AttributionControl({
    compact: false
  });

  const FIT_BOUNDS_SETTINGS = { animate: false, padding: 50, maxZoom: 14 };

  function format(value) {
    if (Array.isArray(value)) {
      const [first, second] = value;
      if (first === -Infinity) {
        return `< ${formatValue(second)}`;
      }
      if (second === Infinity) {
        return `> ${formatValue(first)}`;
      }
      return `${formatValue(first)} - ${formatValue(second)}`;
    }
    return formatValue(value);
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return formatNumber(value);
    }
    return value;
  }

  function formatNumber(value) {
    const log = Math.log10(Math.abs(value));

    if ((log > 4 || log < -2.00000001) && value) {
      return value.toExponential(2);
    }

    if (!Number.isInteger(value)) {
      return value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 3
      });
    }

    return value.toLocaleString();
  }

  function updateViewport(map) {
    function updateMapInfo() {
      const mapInfo$ = document.getElementById('map-info');

      const center = map.getCenter();
      const lat = center.lat.toFixed(6);
      const lng = center.lng.toFixed(6);
      const zoom = map.getZoom().toFixed(2);

      mapInfo$.innerText = `viewport={'zoom': ${zoom}, 'lat': ${lat}, 'lng': ${lng}}`;
    }

    updateMapInfo();

    map.on('zoom', updateMapInfo);
    map.on('move', updateMapInfo); 
  }

  function getBasecolorSettings(basecolor) {
    return {
      'version': 8,
      'sources': {},
      'layers': [{
          'id': 'background',
          'type': 'background',
          'paint': {
              'background-color': basecolor
          }
      }]
    };
  }

  function getImageElement(mapIndex) {
    const id = mapIndex !== undefined ? `map-image-${mapIndex}` : 'map-image';
    return document.getElementById(id);
  }

  function getContainerElement(mapIndex) {
    const id = mapIndex !== undefined ? `main-container-${mapIndex}` : 'main-container';
    return document.getElementById(id);
  }

  function saveImage(mapIndex) {
    const img = getImageElement(mapIndex);
    const container = getContainerElement(mapIndex);

    html2canvas(container)
      .then((canvas) => setMapImage(canvas, img, container));
  }

  function setMapImage(canvas, img, container) {
    const src = canvas.toDataURL();
    img.setAttribute('src', src);
    img.style.display = 'block';
    container.style.display = 'none';
  }

  function createDefaultLegend(layers) {
    const defaultLegendContainer = document.getElementById('default-legend-container');
    defaultLegendContainer.style.display = 'none';

    AsBridge.VL.Legends.layersLegend(
      '#default-legend',
      layers,
      {
        onLoad: () => defaultLegendContainer.style.display = 'unset'
      }
    );
  }

  function createLegend(layer, legendData, layerIndex, mapIndex=0) {
    const element = document.querySelector(`#layer${layerIndex}_map${mapIndex}_legend`);

    if (legendData.prop) {
      const config = { othersLabel: 'Others' };  // TODO: i18n
      const opts = { format, config };

      if (legendData.type.startsWith('size-continuous')) {
        config.samples = 4;
      }

      AsBridge.VL.Legends.rampLegend(
        element,
        layer,
        legendData.prop,
        opts
      );
    }
  }

  /** From https://github.com/errwischt/stacktrace-parser/blob/master/src/stack-trace-parser.js */

  /**
   * This parses the different stack traces and puts them into one format
   * This borrows heavily from TraceKit (https://github.com/csnover/TraceKit)
   */

  const UNKNOWN_FUNCTION = '<unknown>';
  const chromeRe = /^\s*at (.*?) ?\(((?:file|https?|blob|chrome-extension|native|eval|webpack|<anonymous>|\/).*?)(?::(\d+))?(?::(\d+))?\)?\s*$/i;
  const chromeEvalRe = /\((\S*)(?::(\d+))(?::(\d+))\)/;
  const winjsRe = /^\s*at (?:((?:\[object object\])?.+) )?\(?((?:file|ms-appx|https?|webpack|blob):.*?):(\d+)(?::(\d+))?\)?\s*$/i;
  const geckoRe = /^\s*(.*?)(?:\((.*?)\))?(?:^|@)((?:file|https?|blob|chrome|webpack|resource|\[native).*?|[^@]*bundle)(?::(\d+))?(?::(\d+))?\s*$/i;
  const geckoEvalRe = /(\S+) line (\d+)(?: > eval line \d+)* > eval/i;

  function parse(stackString) {
    const lines = stackString.split('\n');

    return lines.reduce((stack, line) => {
      const parseResult =
        parseChrome(line) ||
        parseWinjs(line) ||
        parseGecko(line);

      if (parseResult) {
        stack.push(parseResult);
      }

      return stack;
    }, []);
  }

  function parseChrome(line) {
    const parts = chromeRe.exec(line);

    if (!parts) {
      return null;
    }

    const isNative = parts[2] && parts[2].indexOf('native') === 0; // start of line
    const isEval = parts[2] && parts[2].indexOf('eval') === 0; // start of line

    const submatch = chromeEvalRe.exec(parts[2]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line/column number
      parts[2] = submatch[1]; // url
      parts[3] = submatch[2]; // line
      parts[4] = submatch[3]; // column
    }

    return {
      file: !isNative ? parts[2] : null,
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: isNative ? [parts[2]] : [],
      lineNumber: parts[3] ? +parts[3] : null,
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseWinjs(line) {
    const parts = winjsRe.exec(line);

    if (!parts) {
      return null;
    }

    return {
      file: parts[2],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: [],
      lineNumber: +parts[3],
      column: parts[4] ? +parts[4] : null,
    };
  }

  function parseGecko(line) {
    const parts = geckoRe.exec(line);

    if (!parts) {
      return null;
    }

    const isEval = parts[3] && parts[3].indexOf(' > eval') > -1;

    const submatch = geckoEvalRe.exec(parts[3]);
    if (isEval && submatch != null) {
      // throw out eval line/column and use top-most line number
      parts[3] = submatch[1];
      parts[4] = submatch[2];
      parts[5] = null; // no column when eval
    }

    return {
      file: parts[3],
      methodName: parts[1] || UNKNOWN_FUNCTION,
      arguments: parts[2] ? parts[2].split(',') : [],
      lineNumber: parts[4] ? +parts[4] : null,
      column: parts[5] ? +parts[5] : null,
    };
  }

  function displayError(e) {
    const error$ = document.getElementById('error-container');
    const errors$ = error$.getElementsByClassName('errors');
    const stacktrace$ = document.getElementById('error-stacktrace');

    errors$[0].innerHTML = e.name;
    errors$[1].innerHTML = e.name;
    errors$[2].innerHTML = e.type;
    errors$[3].innerHTML = e.message.replace(e.type, '');

    error$.style.visibility = 'visible';

    const stack = parse(e.stack);
    const list = stack.map(item => {
      return `<li>
      at <span class=&quot;stacktrace-method&quot;>${item.methodName}:</span>
      (${item.file}:${item.lineNumber}:${item.column})
    </li>`;
    });

    stacktrace$.innerHTML = list.join('\n');
  }

  function resetPopupClick(interactivity) {
    interactivity.off('featureClick');
  }

  function resetPopupHover(interactivity) {
    interactivity.off('featureHover');
  }

  function setPopupsClick(map, popup, interactivity, attrs) {
    interactivity.on('featureClick', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function setPopupsHover(map, popup, interactivity, attrs) {
    interactivity.on('featureHover', (event) => {
      updatePopup(map, popup, event, attrs);
    });
  }

  function updatePopup(map, popup, event, attrs) {
    if (event.features.length > 0) {
      let popupHTML = '';
      const layerIDs = [];

      for (const feature of event.features) {
        if (layerIDs.includes(feature.layerId)) {
          continue;
        }
        // Track layers to add only one feature per layer
        layerIDs.push(feature.layerId);

        for (const item of attrs) {
          const variable = feature.variables[item.name];
          if (variable) {
            let value = variable.value;
            value = formatValue(value);

            popupHTML = `
            <span class=&quot;popup-name&quot;>${item.title}</span>
            <span class=&quot;popup-value&quot;>${value}</span>
          ` + popupHTML;
          }
        }
      }

      popup
          .setLngLat([event.coordinates.lng, event.coordinates.lat])
          .setHTML(`<div class=&quot;popup-content&quot;>${popupHTML}</div>`);

      if (!popup.isOpen()) {
        popup.addTo(map);
      }
    } else {
      popup.remove();
    }
  }

  function setInteractivity(map, interactiveLayers, interactiveMapLayers) {
    const interactivity = new carto.Interactivity(interactiveMapLayers);
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    const { clickAttrs, hoverAttrs } = _setInteractivityAttrs(interactiveLayers);

    resetPopupClick(map);
    resetPopupHover(map);

    if (clickAttrs.length > 0) {
      setPopupsClick(map, popup, interactivity, clickAttrs);
    }

    if (hoverAttrs.length > 0) {
      setPopupsHover(map, popup, interactivity, hoverAttrs);
    }
  }

  function _setInteractivityAttrs(interactiveLayers) {
    let clickAttrs = [];
    let hoverAttrs = [];

    interactiveLayers.forEach((interactiveLayer) => {
      interactiveLayer.interactivity.forEach((interactivityDef) => {
        if (interactivityDef.event === 'click') {
          clickAttrs = clickAttrs.concat(interactivityDef.attrs);
        } else if (interactivityDef.event === 'hover') {
          hoverAttrs = hoverAttrs.concat(interactivityDef.attrs);
        }
      });
    });

    return { clickAttrs, hoverAttrs };
  }

  function renderWidget(widget, value) {
    widget.element = widget.element || document.querySelector(`#${widget.id}-value`);

    if (value && widget.element) {
      widget.element.innerText = typeof value === 'number' ? format(value) : value;
    }
  }

  function renderBridge(bridge, widget) {
    widget.element = widget.element || document.querySelector(`#${widget.id}`);

    switch (widget.type) {
      case 'histogram':
        bridge.histogram(widget.element, widget.value, widget.options);
        break;
      case 'category':
        bridge.category(widget.element, widget.value, widget.options);
        break;
      case 'animation':
        widget.options.propertyName = widget.prop;
        bridge.animationControls(widget.element, widget.value, widget.options);
        break;
      case 'time-series':
        widget.options.propertyName = widget.prop;
        bridge.timeSeries(widget.element, widget.value, widget.options);
        break;
    }
  }

  function bridgeLayerWidgets(map, mapLayer, mapSource, widgets) {
    const bridge = new AsBridge.VL.Bridge({
      carto: carto,
      layer: mapLayer,
      source: mapSource,
      map: map
    });

    widgets
      .filter((widget) => widget.has_bridge)
      .forEach((widget) => renderBridge(bridge, widget));

    bridge.build();
  }

  function SourceFactory() {
    const sourceTypes = { GeoJSON, Query, MVT };

    this.createSource = (layer) => {
      return sourceTypes[layer.type](layer);
    };
  }

  function GeoJSON(layer) {
    return new carto.source.GeoJSON(_decodeJSONQuery(layer.query));
  }

  function Query(layer) {
    const auth = {
      username: layer.credentials.username,
      apiKey: layer.credentials.api_key || 'default_public'
    };

    const config = {
      serverURL: layer.credentials.base_url || `https://${layer.credentials.username}.carto.com/`
    };

    return new carto.source.SQL(layer.query, auth, config);
  }

  function MVT(layer) {
    return new carto.source.MVT(layer.query.file, JSON.parse(layer.query.metadata));
  }

  function _decodeJSONQuery(query) {
    return JSON.parse(Base64.decode(query.replace(/b\'/, '\'')));
  }

  const factory = new SourceFactory();

  function initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex) {
    const mapSource = factory.createSource(layer);
    const mapViz = new carto.Viz(layer.viz);
    const mapLayer = new carto.Layer(`layer${layerIndex}`, mapSource, mapViz);
    const mapLayerIndex = numLayers - layerIndex - 1;

    try {
      mapLayer._updateLayer.catch(displayError);
    } catch (e) {
      throw e;
    }

    setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends);
    setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource);

    mapLayer.addTo(map);

    return mapLayer;
  }

  function getInteractiveLayers(layers, mapLayers) {
    const interactiveLayers = [];
    const interactiveMapLayers = [];

    layers.forEach((layer, index) => {
      if (layer.interactivity) {
        interactiveLayers.push(layer);
        interactiveMapLayers.push(mapLayers[index]);
      }
    });

    return { interactiveLayers, interactiveMapLayers };
  }

  function setLayerLegend(layer, mapLayerIndex, mapLayer, mapIndex, hasLegends) {
    if (hasLegends && layer.legend) {
      createLegend(mapLayer, layer.legend, mapLayerIndex, mapIndex);
    }
  }

  function setLayerWidgets(map, layer, mapLayer, mapLayerIndex, mapSource) {
    if (layer.widgets.length) {
      initLayerWidgets(layer.widgets, mapLayerIndex);
      updateLayerWidgets(layer.widgets, mapLayer);
      bridgeLayerWidgets(map, mapLayer, mapSource, layer.widgets);
    }
  }

  function initLayerWidgets(widgets, mapLayerIndex) {
    widgets.forEach((widget, widgetIndex) => {
      const id = `layer${mapLayerIndex}_widget${widgetIndex}`;
      widget.id = id;
    });
  }

  function updateLayerWidgets(widgets, mapLayer) {
    mapLayer.on('updated', () => renderLayerWidgets(widgets, mapLayer));
  }

  function renderLayerWidgets(widgets, mapLayer) {
    const variables = mapLayer.viz.variables;

    widgets
      .filter((widget) => !widget.has_bridge)
      .forEach((widget) => {
        const name = widget.variable_name;
        const value = getWidgetValue(name, variables);
        renderWidget(widget, value);
      });
  }

  function getWidgetValue(name, variables) {
    return name && variables[name] ? variables[name].value : null;
  }

  function setReady(settings) {
    try {
      return settings.maps ? initMaps(settings.maps) : initMap(settings);
    } catch (e) {
      displayError(e);
    }
  }

  function initMaps(maps) {
    return maps.map((mapSettings, mapIndex) => {
      return initMap(mapSettings, mapIndex);
    });
  }

  function initMap(settings, mapIndex) {
    const basecolor = getBasecolorSettings(settings.basecolor);
    const basemapStyle =  BASEMAPS[settings.basemap] || settings.basemap || basecolor;
    const container = mapIndex !== undefined ? `map-${mapIndex}` : 'map';
    const map = createMap(container, basemapStyle, settings.bounds, settings.mapboxtoken);

    if (settings.show_info) {
      updateViewport(map);
    }

    if (settings.camera) {
      map.flyTo(settings.camera);
    }

    return initLayers(map, settings, mapIndex);
  }

  function initLayers(map, settings, mapIndex) {
    const numLayers = settings.layers.length;
    const hasLegends = settings.has_legends;
    const isDefaultLegend = settings.default_legend;
    const isStatic = settings.is_static;
    const layers = settings.layers;
    const mapLayers = getMapLayers(
      layers,
      numLayers,
      hasLegends,
      map,
      mapIndex
    );

    createLegend$1(isDefaultLegend, mapLayers);
    setInteractiveLayers(map, layers, mapLayers);

    return waitForMapLayersLoad(isStatic, mapIndex, mapLayers);
  }

  function waitForMapLayersLoad(isStatic, mapIndex, mapLayers) {
    return new Promise((resolve) => {
      carto.on('loaded', mapLayers, onMapLayersLoaded.bind(
        this, isStatic, mapIndex, mapLayers, resolve)
      );
    });
  }

  function onMapLayersLoaded(isStatic, mapIndex, mapLayers, resolve) {
    if (isStatic) {
      saveImage(mapIndex);
    }

    resolve(mapLayers);
  }

  function getMapLayers(layers, numLayers, hasLegends, map, mapIndex) {
    return layers.map((layer, layerIndex) => {
      return initMapLayer(layer, layerIndex, numLayers, hasLegends, map, mapIndex);
    });
  }

  function setInteractiveLayers(map, layers, mapLayers) {
    const { interactiveLayers, interactiveMapLayers } = getInteractiveLayers(layers, mapLayers);

    if (interactiveLayers && interactiveLayers.length > 0) {
      setInteractivity(map, interactiveLayers, interactiveMapLayers);
    }
  }

  function createLegend$1(isDefaultLegend, mapLayers) {
    if (isDefaultLegend) {
      createDefaultLegend(mapLayers);
    }
  }

  function createMap(container, basemapStyle, bounds, accessToken) {
    const map = createMapboxGLMap(container, basemapStyle, accessToken);

    map.addControl(attributionControl);
    map.fitBounds(bounds, FIT_BOUNDS_SETTINGS);

    return map;
  }

  function createMapboxGLMap(container, style, accessToken) {
    if (accessToken) {
      mapboxgl.accessToken = accessToken;
    }

    return new mapboxgl.Map({
      container,
      style,
      zoom: 9,
      dragRotate: false,
      attributionControl: false
    });
  }

  function init(settings) {
    setReady(settings);
  }

  return init;

}());
</script>
<script>
  const maps = [{&quot;_airship_path&quot;: null, &quot;_carto_vl_path&quot;: null, &quot;basecolor&quot;: &quot;&quot;, &quot;basemap&quot;: &quot;Positron&quot;, &quot;bounds&quot;: [[-77.16358, 38.82744], [-77.04366, 38.90519]], &quot;camera&quot;: {&quot;bearing&quot;: null, &quot;center&quot;: null, &quot;pitch&quot;: null, &quot;zoom&quot;: 11}, &quot;default_legend&quot;: false, &quot;description&quot;: null, &quot;has_legends&quot;: true, &quot;has_widgets&quot;: false, &quot;is_static&quot;: true, &quot;layers&quot;: [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;v8718bf&quot;, &quot;title&quot;: &quot;labels&quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;color&quot;, &quot;title&quot;: &quot;labels&quot;, &quot;type&quot;: &quot;color-category-polygon&quot;}, &quot;query&quot;: &quot;SELECT * FROM \&quot;eschbacher\&quot;.\&quot;demo_augmentation\&quot;&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v8718bf: $labels\ncolor: opacity(ramp(top($labels, 11), bold), 0.9)\nstrokeWidth: ramp(linear(zoom(),2,18),[0.5,1])\nstrokeColor: opacity(#2c2c2c,ramp(linear(zoom(),2,18),[0.2,0.6]))\nfilter: 1\n&quot;, &quot;widgets&quot;: []}], &quot;show_info&quot;: null, &quot;size&quot;: null, &quot;theme&quot;: null, &quot;title&quot;: null, &quot;token&quot;: &quot;&quot;, &quot;viewport&quot;: {&quot;zoom&quot;: 11}}, {&quot;_airship_path&quot;: null, &quot;_carto_vl_path&quot;: null, &quot;basecolor&quot;: &quot;&quot;, &quot;basemap&quot;: &quot;Positron&quot;, &quot;bounds&quot;: [[-77.1560030523493, 38.8341116095337], [-77.0492176172532, 38.8989419527577]], &quot;camera&quot;: {&quot;bearing&quot;: null, &quot;center&quot;: null, &quot;pitch&quot;: null, &quot;zoom&quot;: 11}, &quot;default_legend&quot;: false, &quot;description&quot;: null, &quot;has_legends&quot;: true, &quot;has_widgets&quot;: false, &quot;is_static&quot;: true, &quot;layers&quot;: [{&quot;credentials&quot;: {&quot;api_key&quot;: &quot;01c1be0f2edf4707024f448eaff513552a0b0b4b&quot;, &quot;base_url&quot;: &quot;https://eschbacher.carto.com&quot;, &quot;username&quot;: &quot;eschbacher&quot;}, &quot;interactivity&quot;: [{&quot;attrs&quot;: [{&quot;name&quot;: &quot;v7dd36f&quot;, &quot;title&quot;: &quot;Drop Offs &quot;}], &quot;event&quot;: &quot;hover&quot;}], &quot;legend&quot;: {&quot;description&quot;: &quot;&quot;, &quot;footer&quot;: &quot;&quot;, &quot;prop&quot;: &quot;width&quot;, &quot;title&quot;: &quot;Drop Offs &quot;, &quot;type&quot;: &quot;size-continuous-point&quot;}, &quot;query&quot;: &quot;\nSELECT\n    _ends.num_bike_dropoffs,\n    _starts.num_bike_pickups,\n    abs(_ends.num_bike_dropoffs - _starts.num_bike_pickups) as diff,\n    CASE WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups \u003e 0 THEN 1\n         WHEN _ends.num_bike_dropoffs - _starts.num_bike_pickups = 0 THEN 0\n         ELSE -1 END as diff_sign,\n    _ends.num_bike_dropoffs - _starts.num_bike_pickups as diff_relative,\n    _starts.station_id,\n    row_number() OVER () as cartodb_id,\n    ST_X(_starts.the_geom) as longitude,\n    ST_Y(_starts.the_geom) as latitude,\n    _starts.the_geom,\n    ST_Transform(_starts.the_geom, 3857) as the_geom_webmercator,\n    _ends.day_of_month::numeric as day_of_month\nFROM\n    (SELECT\n      count(u.*) as num_bike_dropoffs,\n      u.end_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id,\n      EXTRACT(DAY FROM end_date) as day_of_month\n    FROM capital_bikeshare_stations_points_arlington as s\n    JOIN capitalbikeshare_tripdata_201907 as u\n    ON u.end_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4, 5) as _ends\nJOIN\n    (SELECT\n      count(u.*) as num_bike_pickups,\n      u.start_station_number::int as station_id,\n      s.the_geom,\n      s.cartodb_id,\n      EXTRACT(DAY FROM start_date) as day_of_month\n    FROM capitalbikeshare_tripdata_201907 as u\n    JOIN capital_bikeshare_stations_points_arlington as s\n    ON u.start_station_number::int = s.gisid::int\n    GROUP BY 2, 3, 4, 5) as _starts\nON _ends.station_id = _starts.station_id and _ends.day_of_month = _starts.day_of_month\n&quot;, &quot;type&quot;: &quot;Query&quot;, &quot;viz&quot;: &quot;@v7dd36f: $diff_sign * $diff\ncolor: opacity(ramp($diff_sign, antique), 0.8)\nwidth: ramp(linear(sqrt($diff), sqrt(globalMin($diff)), sqrt(globalMax($diff))), [2, 40])\nstrokeWidth: ramp(linear(zoom(),0,18),[0,1])\nstrokeColor: opacity(#222,ramp(linear(zoom(),0,18),[0,0.6]))\n&quot;, &quot;widgets&quot;: []}], &quot;show_info&quot;: null, &quot;size&quot;: null, &quot;theme&quot;: null, &quot;title&quot;: null, &quot;token&quot;: &quot;&quot;, &quot;viewport&quot;: {&quot;zoom&quot;: 11}}];
const show_info = 'None' === 'true';
const is_static = 'true' === 'true';

init({
  show_info,
  is_static,
  maps
});
</script>
</html>
">

</iframe>




```python

```