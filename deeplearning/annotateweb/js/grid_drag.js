$(document).ready(function(){

  $("#grid").mousedown(function (e) {

        $(".ghost-select").addClass("ghost-active");
        $(".ghost-select").css({
            'left': e.pageX,
            'top': e.pageY
        });

        initialW = e.pageX;
        initialH = e.pageY;

        console.log("attach handler");

        $(document).bind("mouseup", selectElements);
        $(document).bind("mousemove", openSelector);

    });


});

function selectElements(e) {
    $("#score>span").text('0');
    $(document).unbind("mousemove", openSelector);
    $(document).unbind("mouseup", selectElements);
    var maxX = 0;
    var minX = 5000;
    var maxY = 0;
    var minY = 5000;
    var totalElements = 0;
    var elementArr = new Array();
    $(".elements").each(function () {
        var aElem = $(".ghost-select");
        var bElem = $(this);
        var result = doObjectsCollide(aElem, bElem);

        console.log(result);
        if (result == true) {
          $("#score>span").text( Number($("#score>span").text())+1 );
        }
    });

    $(".ghost-select").removeClass("ghost-active");
    $(".ghost-select").width(0).height(0);

    ////////////////////////////////////////////////

}

function openSelector(e) {
    var w = Math.abs(initialW - e.pageX);
    var h = Math.abs(initialH - e.pageY);

    $(".ghost-select").css({
        'width': w,
        'height': h
    });
    if (e.pageX <= initialW && e.pageY >= initialH) {
        $(".ghost-select").css({
            'left': e.pageX
        });
    } else if (e.pageY <= initialH && e.pageX >= initialW) {
        $(".ghost-select").css({
            'top': e.pageY
        });
    } else if (e.pageY < initialH && e.pageY < initialW) {
        $(".ghost-select").css({
            'left': e.pageX,
            "top": e.pageY
        });
    }
}


function doObjectsCollide(a, b) { // a and b are your objects
    //console.log(a.offset().top,a.position().top, b.position().top, a.width(),a.height(), b.width(),b.height());
    var aTop = a.offset().top;
    var aLeft = a.offset().left;
    var bTop = b.offset().top;
    var bLeft = b.offset().left;

    return !(
        ((aTop + a.height()) < (bTop)) ||
        (aTop > (bTop + b.height())) ||
        ((aLeft + a.width()) < bLeft) ||
        (aLeft > (bLeft + b.width()))
    );
}
