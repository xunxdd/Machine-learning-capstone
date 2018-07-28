$(function () {
    $.getJSON( "/static/data.json", function( data ) {
        var normal = data.filter((x) => x.label === "0");
        var bacterial = data.filter((x) => x.label === "1");
        var viral = data.filter((x) => x.label === "2");
        outSection(normal, $('#normal-xray-section'));
        outSection(bacterial, $('#bacterial-xray-section'));
        outSection(viral, $('#viral-xray-section'));
    });

    function outSection(data, target) {
        $.each(data, function (index, val) {
            if (index < 6) {
                html = outputImg(index, val.img);
                target.append(html);
            }
        });
    }

    function outputImg(index, img) {
        return `<div class="col-sm-6 col-md-4">
                   <a class="lightbox" href="${img}" target="_blank">
                        <img src="${img}" width="300" height="300"/>
                    </a>
               </div>`;
    }
});