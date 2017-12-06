from beatsearch.app.control import BSController
from beatsearch.app.view import BSApp

if __name__ == "__main__":
    controller = BSController()
    controller.set_corpus("./data/rhythms.pkl")
    app = BSApp(controller, transport_frame_cls=None)
    app.mainloop()
